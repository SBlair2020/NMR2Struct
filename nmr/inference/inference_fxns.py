import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from torch import Tensor

def infer_basic_model(model: nn.Module, 
                      batch: torch.Tensor, 
                      opts: Optional[dict] = None,
                      device: torch.device = None) -> torch.Tensor:
    """Generate prediction for models that take an input and generate the output in one forward pass
    Args: 
        model: The model to use for inference
        input: The input to the model
        opts: Options to pass to the model as a dictionary, should specify gradient tracking 
            behavior through value 'track_gradients'. If a dictionary is not provided,
            the default behavior is to track gradients
    """
    x, y = batch
    target = y[0]
    #Need additional logic around gradient tracking for modules associated with the 
    #   transformer because it seems behavior can change depending on the no_grad() context
    if opts is None:
        track_gradients = False #Default option
    elif (opts is not None) and ('track_gradients' in opts):
        track_gradients = opts['track_gradients']
    if track_gradients:
        output = model(x, y)
    else:
        with torch.no_grad():
            output = model(x)
    #Also save x[1] which is the set of SMILES strings
    #   Note that even for a batch size of 1, the batch smiles element
    #   returned by a dataloader is a tuple of the form (str,) which converts
    #   correctly to [str] when using list(). It does not cause the string
    #   to break apart into a list of characters.
    return [(
        target.detach().cpu().numpy(), 
        output.detach().cpu().numpy(),
        list(x[1])
    )]

def get_top_k_sample_batched(k_val: int | float , 
                             character_probabilities: Tensor) -> Tensor:
    """
    Generates the next character using top-k sampling scheme.

    In top-k sampling, the probability mass is redistributed among the
    top-k next tokens, where k is a hyperparameter. Once redistributed, 
    the next token is sampled from the top-k tokens.
    """
    top_values, top_indices = torch.topk(character_probabilities, k_val, sorted = True)
    #Take the sum of the top probabilities and renormalize
    tot_probs = top_values / torch.sum(top_values, dim = -1).reshape(-1, 1)
    #Sample from the top k probabilities. This represents a multinomial distribution
    try:
        assert(torch.allclose(torch.sum(tot_probs, dim = -1), torch.tensor(1.0)))
    except:
        print("Probabilities did not pass allclose check!")
        print(f"Sum of probs is {torch.sum(tot_probs)}")
    selected_index = torch.multinomial(tot_probs, 1)
    #For gather to work, both tensors have to have the same number of dimensions:
    if len(top_indices.shape) != len(selected_index.shape):
        top_indices = top_indices.reshape(selected_index.shape[0], -1)
    output = torch.gather(top_indices, -1, selected_index)
    return output

def infer_transformer_model(model: nn.Module, 
                        batch: torch.Tensor, 
                        opts: dict,
                        device: torch.device = None) -> list[tuple[str, list[str]]] | list[tuple[np.ndarray, list[np.ndarray]]]:
    """Generates a prediction for the input using sampling over a transformer model
    Args:
        model: The model to use for inference
        batch: The input to the model
        opts: Options to pass to the model as a dictionary
        device: The device to use for inference
    
    The opts dictionary should contain the following additional arguments:
        'num_pred_per_tgt' (int): The number of predictions to generate for each input
        'sample_val' (int or float): The sampling value for the model, e.g. the number of values to use for
            top-k sampling
        'stop_token' (int): The stop token to use for the model
        'start_token' (int): The start token to use for the model
        'track_gradients' (bool): Whether to track gradients during inference. This is because the transformer
            is known to misbehave if gradient tracking is disabled in certain cases
        'alphabet' (str): Path to a file containing the alphabet for the model to use in decoding
        'decode' (bool): Whether to decode the output indices of the model against the provided alphabet
    """
    x, y = batch
    curr_batch_predictions = []
    effective_bsize = x[0].shape[0]
    targets = y[1]
    smiles = x[1]
    
    num_pred_per_tgt = opts['num_pred_per_tgt']
    sample_val = opts['sample_val']
    stop_token = opts['tgt_stop_token']
    start_token = opts['tgt_start_token']
    track_gradients = opts['track_gradients']
    alphabet = np.load(opts['alphabet'], allow_pickle=True)
    decode = opts['decode']

    for _ in range(num_pred_per_tgt):

        working_x = x[0].clone()
        working_y = torch.tensor([start_token] * effective_bsize).reshape(effective_bsize, 1).to(device)

        completed_structures = [None] * effective_bsize
        index_mapping = torch.arange(effective_bsize, device = device, dtype = torch.long)
        all_structures_completed = False
        iter_counter = 0

        while not all_structures_completed:
            if (iter_counter % 10 == 0):
                print(f"On iteration {iter_counter}")
            
            if track_gradients:
                next_pos = model((working_x, None), (working_y, None)).detach()
            else:
                with torch.no_grad():
                    next_pos = model((working_x, None), (working_y, None))
            
            next_val = next_pos[:, -1, :]
            char_probs = torch.nn.functional.softmax(next_val, dim = -1)
            selected_indices = get_top_k_sample_batched(sample_val, char_probs)
            concatenated_results = torch.cat((working_y, selected_indices), dim = -1)
            stop_token_mask = concatenated_results[:, -1] == stop_token
            comp_structs = concatenated_results[stop_token_mask]
            comp_inds = index_mapping[stop_token_mask]
            for i, ind in enumerate(comp_inds):
                completed_structures[ind] = comp_structs[i].detach().cpu().numpy()
            working_y = concatenated_results[~stop_token_mask]
            working_x = working_x[~stop_token_mask]
            index_mapping = index_mapping[~stop_token_mask]
            if working_y.shape[-1] > 1000:
                working_y = torch.cat((working_y,
                                       torch.tensor([stop_token] * working_y.shape[0]).reshape(-1, 1).to(device)), 
                                       dim = -1)
                for j, ind in enumerate(index_mapping):
                    completed_structures[ind] = working_y[j].detach().cpu().numpy()
                all_structures_completed = True
            if len(working_y) == 0:
                all_structures_completed = True
            iter_counter += 1
        for elem in completed_structures:
            assert(elem is not None)
        if decode:
            generated_smiles = [''.join(np.array(alphabet)[elem[1:-1].astype(int)]) for elem in completed_structures]
            curr_batch_predictions.append(generated_smiles)
        else:
            curr_batch_predictions.append(completed_structures)
        
    #Final processing
    # print(targets.shape[0])
    # print(len(curr_batch_predictions))
    # print(effective_bsize)
    assert(targets.shape[0] == len(curr_batch_predictions[0]) == effective_bsize)
    generated_predictions = []
    for i in range(effective_bsize):
        #TODO: Think, is this the best way to represent output predictions for each batch as 
        #   tuples of (target, [pred1, pred2, ...])?
        generated_predictions.append((
            smiles[i] if decode else targets[i].detach().cpu().numpy(),
            list(curr_batch_predictions[j][i] for j in range(num_pred_per_tgt)),
            smiles[i]
        ))
    return generated_predictions
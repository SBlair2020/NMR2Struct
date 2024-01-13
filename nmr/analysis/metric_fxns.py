import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch
from rdkit import Chem as chem
from .util import mols_to_labels

### Parallelizable fxns ###
def calc_sub_BCE(true_labels: np.ndarray, 
                 candidates_labels: np.ndarray, 
                 axis: int = -1) -> np.ndarray:
    """
    true_labels, shape(s,)
    candidates_labels, shape(n,s)
    """
    probs = true_labels*candidates_labels + (1-true_labels)*(1-candidates_labels)
    result = - np.log(probs + np.finfo(float).eps)
    if axis != None:
        result = result.mean(axis=axis)
    return result

def compute_molecule_BCE(predictions: list[list[chem.Mol]],
                         targets: list[str],
                         substructures: list[chem.Mol]) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """Computes molecule BCEs for a set of sanitized predictions and targets
    Args:
        predictions: list of list of sanitized and valid predictions
        targets: list of sanitized and valid targets
        substructures: List of substructure molecules generated from the substructure SMART strings
    """
    assert(len(predictions) == len(targets))
    targets = []
    preds_and_losses = []
    for i_str, targ in enumerate(targets):
        targ_mol = chem.molFromSmiles(targ)
        targets.append(targ)
        curr_pred = predictions[i_str]
        true_subs = mols_to_labels([targ_mol], substructures)[0]
        current_profs = mols_to_labels(curr_pred, substructures)
        losses = calc_sub_BCE(true_subs, current_profs)
        curr_pred_strs = []
        for p in curr_pred:
            try:
                curr_pred_strs.append(chem.CanonSmiles(chem.MolToSmiles(p)))
            except:
                curr_pred_strs.append(chem.MolToSmiles(p))
        pred_losses = list(zip(curr_pred_strs, losses))
        #Sort by loss value so best prediction is first
        pred_losses.sort(key = lambda x: x[1])
        preds_and_losses.append(pred_losses)
    return targets, preds_and_losses

### Substructure metrics on collated data ###
def calc_loss_per_sub(predictions: np.ndarray,
                      targets: np.ndarray) -> np.ndarray:
    '''
    Legacy function inherited from old notebooks that computes the 
    binary cross entropy loss for each input WITHOUT reduction

    predictions shape: (n_samples, n_substructures)
    targets shape: (n_samples, n_substructures)
    '''
    loss = nn.BCELoss(reduction = 'none')
    loss_arr = loss(predictions.T, targets.T).numpy()
    #Averaging over the last axis for consistency with Keras
    loss_arr = np.mean(loss_arr, axis = -1)
    assert(loss_arr.shape == (957,))
    return loss_arr

def get_acc_per_sub(predictions: np.ndarray,
                    targets: np.ndarray) -> np.ndarray:
    '''
    Computes accuracy per substructure
    predictions shape: (n_samples, n_substructures)
    targets shape: (n_samples, n_substructures)
    '''
    m = targets.shape[0]
    correct = targets == (predictions.round())
    accuracy = torch.sum(correct, axis = 0) / m
    return accuracy.numpy()

def compute_roc_auc_score(predictions: np.ndarray,
                          targets: np.ndarray) -> float:
    '''This method expects flattened predictions and targets
    predictions shape: (n_samples * n_substructures)
    targets shape: (n_samples * n_substructures)
    '''
    return metrics.roc_auc_score(targets, predictions)

def compute_precision_recall_auc_scores(predictions: np.ndarray,
                          targets: np.ndarray) -> float:
    '''This method expects flattened predictions and targets
    predictions shape: (n_samples * n_substructures)
    targets shape: (n_samples * n_substructures)
    '''
    precision, recall, _ = metrics.precision_recall_curve(
            targets, predictions)
    prc_auc_score = metrics.auc(recall, precision)
    return precision, recall, prc_auc_score

def compute_fscore(predictions: np.ndarray,
                   targets: np.ndarray) -> float:
    '''This method expects flattened predictions and targets
    predictions shape: (n_samples * n_substructures)
    targets shape: (n_samples * n_substructures)
    '''
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            targets, predictions >= 0.5, average = 'binary')
    return fscore

def compute_exact_seq_match(predictions: np.ndarray,
                            targets: np.ndarray) -> float:
    '''
    predictions shape: (n_samples, n_substructures)
    targets shape: (n_samples, n_substructures)
    '''
    num_match = 0
    for i in range(predictions.shape[0]):
        rounded_preds = (predictions[i] >= 0.5).astype(int)
        if np.allclose(rounded_preds, targets[i]):
            num_match += 1
    return (num_match / predictions.shape[0]) * 100

def compute_total_substruct_metrics(predictions: np.ndarray,
                                    targets: np.ndarray) -> dict:
    '''
    Computes the following and generates a dictionary of results:
        - substructure losses
        - substructure accuracies 
        - precision
        - recall
        - fscore
        - roc_auc_score
        - prc_auc_score
        - exact sequence match percent
    '''
    flat_tgt, flat_pred = targets.flatten(), predictions.flatten()
    substruct_losses = calc_loss_per_sub(predictions, targets)
    substruct_accs = get_acc_per_sub(predictions, targets)
    roc_auc_score = compute_roc_auc_score(flat_pred, flat_tgt)
    precision, recall, prc_auc_score = compute_precision_recall_auc_scores(
            flat_pred, flat_tgt)
    fscore = compute_fscore(flat_pred, flat_tgt)
    exact_seq_match_percent = compute_exact_seq_match(predictions, targets)

    return {
        'substruct_losses': substruct_losses,
        'substruct_accs': substruct_accs,
        'roc_auc_score': roc_auc_score,
        'precision': precision,
        'recall': recall,
        'prc_auc_score': prc_auc_score,
        'fscore': fscore,
        'exact_seq_match_percent': exact_seq_match_percent
    }

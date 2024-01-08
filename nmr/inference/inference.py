import torch
from torch import nn
from typing import Callable
import nmr.inference.inference_fxns as inference_fxns


def run_inference(model: nn.Module,
                  dataset: torch.utils.data.Dataset,
                  pred_gen_fn: Callable[[nn.Module, torch.Tensor, dict], torch.Tensor],
                  pred_gen_opts: dict, 
                  batch_size: int,
                  write_freq: int = 100,
                  dtype: torch.dtype = None,
                  device: torch.device = None):
    """Run inference on a trained model and generate predictions from the examples in dataset

    Args:
        model: A prepared instance of the trained model
        dataset: The dataset containing the examples to generate predictions for
        pred_gen_fn: A function that generates predictions from the model, input, and options
        pred_gen_opts: Options to pass to the prediction generator function
        batch_size: The batch size to use for inference
        dtype: The datatype to use for inference
        device: The device to use for inference
    """
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)
    predictions = []
    pred_gen_fn = getattr(inference_fxns, pred_gen_fn)
    for ibatch, batch in enumerate(dataloader):
        if (ibatch % write_freq == 0):
            print(f"On batch {ibatch}")
        batch_prediction = pred_gen_fn(model, batch, pred_gen_opts)
        predictions.append(batch_prediction)
    return predictions
import nmr.models
import torch
import torch.nn as nn
from typing import Any

def create_model(model_args: dict, dtype: torch.dtype, device: torch.device) -> nn.Module:
    """Creates the model by passing argument dictoinaries into fetched constructors
    Args:
        model_args: The dictionary of model arguments, possibly a highly nested structure
        dtype: The datatype to use for the model
        device: The device to use for the model
    """
    model_base = getattr(nmr.models, model_args['model_type'])
    model_config = model_args['model_args']
    model = model_base(dtype=dtype,
                       device=device,
                       **model_config)
    if model_args['load_model'] is not None:
        ckpt = torch.load(model_args['load_model'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    #Freeze requisite components
    model.freeze()

    #Initialize the weights if not loading a model
    if model_args['load_model'] is None:
        model.initialize_weights()

    #Update the model args
    model_args['model_args'] = model_config

    return model, model_args

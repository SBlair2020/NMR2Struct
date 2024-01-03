import nmr.models
import torch
import torch.nn as nn
from typing import Any

def specific_update(mapping: dict[str, Any], update_map: dict[str, Any]) -> dict[str, Any]:
    """Recursively update keys in a mapping with the values specified in update_map"""
    mapping = mapping.copy()
    for k, v in mapping.items():
        #Give precedence to existing parameter settings
        if (k in update_map) and (not isinstance(v, dict)) and (v is None):
            mapping[k] = update_map[k]
        elif isinstance(v, dict):
            mapping[k] = specific_update(v, update_map)
    return mapping

def create_model(model_args: dict, dtype: torch.dtype, device: torch.device, addn_opts: dict) -> nn.Module:
    """Creates the model by passing argument dictoinaries into fetched constructors
    Args:
        model_args: The dictionary of model arguments, possibly a highly nested structure
        dtype: The datatype to use for the model
        device: The device to use for the model
        addn_opts: The dictionary of sizes for the dataset, with keys 'source_size' and 'target_size', and 
            values for the control tokens in both source and target
    """
    model_base =  getattr(nmr.models, model_args['model_type'])
    model_config = model_args['model_args']
    #Inject size + token information (if applicable)
    model_config = specific_update(model_config, addn_opts)
    model = model_base(dtype=dtype,
                       device=device,
                       **model_config)
    if model_args['load_model'] is not None:
        ckpt = torch.load(model_args['load_model'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    #Freeze requisite components
    model.freeze()

    return model

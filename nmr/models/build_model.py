import models
import torch
import torch.nn as nn

def create_model(model_args: dict, dtype: torch.dtype, device: torch.device) -> nn.Module:
    model_base =  getattr(models, model_args['model_type'])
    model_config = model_args['model_args']
    model = model_base(dtype=dtype,
                       device=device,
                       **model_config)
    if model_args['load_model'] is not None:
        ckpt = torch.load(model_args['load_model'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    #Freeze requisite components
    model.freeze()

    return model

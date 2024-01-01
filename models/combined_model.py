import torch
from torch import nn, Tensor
from typing import Tuple, Callable, Optional, Any, Union
import models

class CombinedModel(nn.Module):
    """Example model wrapper for the combined model"""
    model_id = 'combined'

    def __init__(self, model_1: str, model_2: str, 
                 model_1_args: dict, model_2_args: dict,
                 model_1_freeze_components: Optional[list] = None,
                 model_2_freeze_components: Optional[list] = None,
                 expand_dims: bool = True,
                 device: torch.device = None, 
                 dtype: torch.dtype = torch.float):
        """Constructor for combined model built from two sub models
        
        Args:
            model_1: The name of the first sub model
            model_2: The name of the second sub model
            model_1_args: The kwargs for the first sub model constructor
            model_2_args: The kwargs for the second sub model constructor
            model_1_freeze_components: List of component names to freeze for model_1
            model_2_freeze_components: List of component names to freeze for model_2
            expand_dims: Whether to expand the LAST dimension of the output of model_1 before 
                passing into model_2. Default is True
            device: Model device. Default is None
            dtype: Model datatype. Default is torch.float
        """
        super().__init__()
        self.model_1 = getattr(models, model_1)(
            freeze_components = model_1_freeze_components,
            **model_1_args
        )
        self.model_2 = getattr(models, model_2)(
            freeze_components = model_2_freeze_components, 
            **model_2_args
        )
        self.expand_dims = expand_dims
    
    def forward(self, mod_1_input: Tensor, mod_2_input: Optional[Tensor]) -> Tensor:
        '''
        Args:
            mod_1_input: Tensor for input to model_1
            mod_2_input: Additional optional tensor inputs for model_2 
        '''
        mod_1_output = self.model_1(mod_1_input)
        if self.expand_dims:
            mod_1_output = mod_1_output.unsqueeze(-1)
        if mod_2_input is not None:
            final_out = self.model_2(mod_1_output, mod_2_input)
        else:
            final_out = self.model_2(mod_1_output)
        return final_out
    
    def get_loss(self, 
                 x: tuple[Tensor, tuple],
                 y: tuple[Tensor, Tensor] | tuple[Tensor],
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        inp, smiles = x
        assert(isinstance(y, tuple))
        if len(y) == 2:
            shifted_y, full_y = y
        elif len(y) == 1:
            full_y, = y
            shifted_y = None
        pred = self.forward(inp, shifted_y)
        if shifted_y is not None:
            pred = pred.permute(0, 2, 1)
        loss = loss_fn(pred, full_y.to(self.device))
        return loss
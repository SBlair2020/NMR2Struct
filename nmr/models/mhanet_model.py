from nmr.networks import mhanet, forward_fxns, embeddings
import torch
from torch import nn, Tensor
from typing import tuple, Callable, Optional, Any

class MHANetModel(nn.Module):

    model_id = 'MHANet'

    def __init__(self, 
                 src_embed: str,
                 positional_encoding: Optional[str],
                 forward_network: str, 
                 src_pad_token: int,
                 src_forward_function: str,
                 d_model: int,
                 d_out: int,
                 d_feedforward: int, 
                 n_heads: int, 
                 max_seq_len: int,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 freeze_components: Optional[list[str]] = None):
        super().__init__()
        src_embed_module = getattr(embeddings, src_embed)
        positional_encoding_module = getattr(mhanet, positional_encoding) if positional_encoding is not None else None
        forward_network_module = getattr(mhanet, forward_network)  
        src_forward_function = getattr(forward_fxns, src_forward_function)
        
        self.network = mhanet.MHANet(
            src_embed_module,
            positional_encoding_module,
            forward_network_module,
            src_pad_token,
            src_forward_function,
            d_model,
            d_out,
            d_feedforward,
            n_heads,
            max_seq_len,
            device,
            dtype
        )
        self.freeze_components = freeze_components

    def freeze(self) -> None:
        """Disables gradients for specific components of the network
        
        Args:
            components: A list of strings corresponding to the model components
                to freeze, e.g. src_embed, tgt_embed.
        """
        #TODO: This will need careful testing
        if self.freeze_components is not None:
            for component in self.freeze_components:
                if hasattr(self.network, component):
                    for param in getattr(self.network, component).parameters():
                        param.requires_grad = False
    
    def forward(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        self.network(x)

    def get_loss(self, 
                 x: tuple[Tensor, tuple[str]], 
                 y: tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return self.network.get_loss(x, y, loss_fn) 


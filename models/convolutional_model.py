from networks import convolutional
import torch
from torch import nn, Tensor
from typing import Tuple, Callable, Optional

class ConvolutionalModel(nn.Module):
    """ Example model wrapper for convolutional network """

    def __init__(self, n_spectral_features: int, n_Cfeatures: int, n_molfeatures: int, n_substructures: int,
                 freeze_components: Optional[list] = None,
                 device: torch.device = None, dtype: torch.dtype = torch.float):
        """Constructor for convolutional network model using the NMRConvNet from networks
        Args:
            n_spectral_features: The number of spectral features, i.e. 28000
            n_Cfeatures: The number of CNMR features, i.e. 40
            n_molfeatures: The number of chemical formula features, i.e. 5
            n_substructures: The number of substructures to predict for. This is used for 
                constructing a single linear head for each substructure
            freeze_components: List of component names to freeze
            device: Model device. Default is None
            dtype: Model datatype. Default is torch.float
        """
        super().__init__()
        self.network = convolutional.NMRConvNet(n_spectral_features,
                                                n_Cfeatures,
                                                n_molfeatures,
                                                n_substructures,
                                                dtype,
                                                device)
        if freeze_components is not None:
            self.freeze(freeze_components)
    
    def freeze(self, components: list[str]) -> None:
        """Disables gradients for specific components of the network
        
        Args:
            components: A list of strings corresponding to the model components
                to freeze, e.g. src_embed, tgt_embed.
        """
        #TODO: This will need careful testing
        for component in components:
            if hasattr(self.network, component):
                for param in getattr(self.network, component).parameters():
                    param.requires_grad = False
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, 1, seq_len)
        """
        return self.network(x)
    
    def get_loss(self,
                 x: Tuple[Tensor, Tuple], 
                 y: Tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return self.network.get_loss(x, y, loss_fn)
        




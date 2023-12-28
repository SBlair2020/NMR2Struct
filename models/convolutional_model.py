from networks import convolutional
import torch
from torch import nn, Tensor
from typing import Tuple, Callable

class ConvolutionalModel(nn.Module):
    """ Example model wrapper for convolutional network """

    def __init__(self, n_spectral_features: int, n_Cfeatures: int, n_molfeatures: int, n_substructures: int,
                 dtype: torch.dtype = torch.float, device: torch.device = None):
        """Constructor for convolutional network model using the NMRConvNet from networks
        Args:
            n_spectral_features: The number of spectral features, i.e. 28000
            n_Cfeatures: The number of CNMR features, i.e. 40
            n_molfeatures: The number of chemical formula features, i.e. 5
            n_substructures: The number of substructures to predict for. This is used for 
                constructing a single linear head for each substructure
            dtype: Model datatype. Default is torch.float
            device: Model device. Default is None
        """
        super().__init__()
        self.network = convolutional.NMRConvNet(n_spectral_features,
                                                n_Cfeatures,
                                                n_molfeatures,
                                                n_substructures,
                                                dtype,
                                                device)
    
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
        




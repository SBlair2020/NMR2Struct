import torch
from torch import Tensor
from torch import nn
import math

class ProbabilityEmbedding(nn.Module):
    """ MLP based connection between substructure predictions and transformer. """

    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (batch_size, seq_len, 1)
        '''
        return self.layers(x)
    
class SingleLinear(nn.Module):
    """Single linear layer to connect value to correct model dimensionality"""
    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.Linear(1, d_model)
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class MatrixScaleEmbedding(nn.Module):
    """ Broadcast matrix multiplication connection between substructure predictions and transformer. """

    def __init__(self, d_model: int, n_substructures: int = 957):
        super().__init__()
        self.n_substructures = n_substructures 
        self.layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_substructures, d_model)))
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (batch_size, seq_len, 1)
        '''
        return x * self.layers
    
class NMRContinuousEmbedding(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 1):
        '''
        For implementation simplicity, only use one head for now,
            extending to multiple heads is trivial.
        '''
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(2, d_model // num_heads) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        '''
        x: Tensor, shape (batch_size, seq_len, 2)
        Returns the embedded tensor (batch_size, seq_len, d_model)
        '''
        out = [head(x) for head in self.heads]
        return torch.cat(out, dim = -1)
    
class ConvolutionalEmbedding(nn.Module):
    
    def __init__(self, 
                 d_model: int,
                 pool_variation: str = 'max',
                 pool_size_1: int = 12,
                 out_channels_1: int = 64,
                 kernel_size_1: int = 5,
                 pool_size_2: int = 20,
                 out_channels_2: int = 128,
                 kernel_size_2: int = 9,
                 add_pos_encode: bool = True):
        """Construct features over the spectrum using the same convolutional heads as 
        the convolutional neural network. The convolutional head involves 
        two 1D convolutions interspersed with max pooling. The channel dimensionalities
        are tunable, but default is out_channels_one = 64, out_channels_two = 128. 
        Convolution strides are 1, padding is 'valid' (no padding), and the activation
        function is ReLU. 

        Args:
            d_model: Model dimensionality for downstream transformer
            pool_variation: The type of pooling to use, either 'max' or 'avg' where
                'max' is max pooling and 'avg' is average pooling, both 1D variants
            pool_size_1/2: Size and stride for the respective max pooling layer
            out_channels_1/2: Number of output channels after the respective convolutional layer
            kernel_size_1/2: Kernel size for the respective convolutional layer
            add_pos_encode: Whether to add a positional encoding to the output of this source 
                embedding.

        Notes: 
            Original architectures:
                conv1: Kernel size = 5, Filters (out channels) = 64, in channels = 1
                pool1: Max pool of size 12 with stride 12
                conv2: Kernel size of 9, Filters (out channels) = 128, in channels = 64
                pool2: Max pool of size 20 with stride 20
        """
        super().__init__()
        self.n_spectral_features = 28000
        self.n_Cfeatures = 40
        self.c_embed = nn.Embedding(self.n_Cfeatures + 1, d_model, padding_idx=0)
        self.post_conv_transform = nn.Linear(out_channels_2, d_model)

        if pool_variation == 'max':
            self.pool1 = nn.MaxPool1d(pool_size_1)
            self.pool2 = nn.MaxPool1d(pool_size_2)
        elif pool_variation == 'avg':
            self.pool1 = nn.AvgPool1d(pool_size_1)
            self.pool2 = nn.AvgPool1d(pool_size_2)

        self.conv1 = nn.Conv1d(1, 
                               out_channels_1,
                               kernel_size_1, 
                               stride = 1, 
                               padding = 'valid')
        self.conv2 = nn.Conv1d(out_channels_1, 
                               out_channels_2, 
                               kernel_size_2, 
                               stride = 1, 
                               padding = 'valid')
        self.relu = nn.ReLU()
        self.h_spectrum_final_seq_len = self._compute_final_seq_len(
            self.n_spectral_features,
            [(kernel_size_1, pool_size_1, pool_variation), 
             (kernel_size_2, pool_size_2, pool_variation)]
        )
        self.add_pos_encoder = add_pos_encode

        print("Final sequence length after conv embedding:")
        print(self.h_spectrum_final_seq_len)
    
    #From https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    def _calculate_dim_after_conv(self, 
                                  L_in: int,
                                  kernel: int,
                                  padding: int,
                                  dilation: int,
                                  stride: int) -> int:
        numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
        return math.floor(
            (numerator/stride) + 1
        )
    #From https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
    # and https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    def _calculate_dim_after_pool(self,
                                  pool_variation: str,
                                  L_in: int,
                                  kernel: int,
                                  padding: int,
                                  dilation: int,
                                  stride: int) -> int:
        if pool_variation == 'max':
            numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
            return math.floor(
                (numerator/stride) + 1
            )
        elif pool_variation == 'avg':
            numerator = L_in + (2 * padding) - kernel
            return math.floor(
                (numerator/stride) + 1
            )
    
    def _compute_final_seq_len(self,
                               L_in: int,
                               block_args: list[tuple[int]]) -> int:
        '''Computes the final sequence after a series of convolution + pooling operations
        Args:
            L_in: The initial sequence length
            block_args: A list of tuples, each containing the:
                convolution kernel
                pool kernel
                pooling_variation
        
        This function assumes:
            padding = 0
            dilation = 1
            stride = 1 for conv, stride = pool_size for pool
        '''
        L_final = L_in
        for conv_kernel, pool_kernel, pool_variation in block_args:
            L_final = self._calculate_dim_after_conv(L_final, conv_kernel, 0, 1, 1)
            L_final = self._calculate_dim_after_pool(pool_variation, L_final, pool_kernel, 0, 1, pool_kernel)
        return L_final

    def _separate_spectra_components(self, x: Tensor):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)
        spectral_x = x[:, :, :self.n_spectral_features]
        cnmr_x = x[:, :, self.n_spectral_features:self.n_spectral_features + self.n_Cfeatures]
        mol_x = x[:, :, self.n_spectral_features + self.n_Cfeatures:]
        return spectral_x, cnmr_x, mol_x
    
    def _embed_cnmr(self, cnmr: Tensor):
        """Embeds the binary tensor into a continuous space
        Convert to 1-indexed indices, pad with 0
        """
        assert(cnmr.shape[-1] == self.n_Cfeatures)
        if cnmr.ndim == 3:
            cnmr = cnmr.squeeze(1)
        indices = torch.arange(0, self.n_Cfeatures) + 1
        indices = indices.to(cnmr.dtype).to(cnmr.device)
        cnmr = cnmr * indices
        cnmr[cnmr == 0] = 100
        cnmr = torch.sort(cnmr).values
        cnmr[cnmr == 100] = 0
        #print(torch.max(cnmr))
        #print(torch.min(cnmr))
        return self.c_embed(cnmr.long())

    def _embed_spectra(self, spectra: Tensor):
        assert spectra.ndim == 3
        spectra = self.conv1(spectra)
        spectra = self.relu(spectra)
        spectra = self.pool1(spectra)
        spectra = self.conv2(spectra)
        spectra = self.relu(spectra)
        spectra = self.pool2(spectra) #(N, E, T)
        spectra = torch.transpose(spectra, 1, 2) #(N, T, E)
        return self.post_conv_transform(spectra)

    def forward(self, x):
        spectra, cnmr, mol = self._separate_spectra_components(x)   
        cnmr_embed = self._embed_cnmr(cnmr)
        spectra_embed = self._embed_spectra(spectra)
        #print(spectra_embed.shape)
        #print(cnmr_embed.shape)
        return torch.cat((spectra_embed, cnmr_embed), dim = 1)

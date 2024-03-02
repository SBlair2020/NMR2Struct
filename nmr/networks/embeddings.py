import torch
from torch import Tensor
from torch import nn

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
    
    def __init__(self, d_model: int):
        """Construct features over the spectrum using the same convolutional heads as 
        the convolutional neural network

        For simplicty, a lot of the parameters related to the spectrum are hard-coded
        """
        super().__init__()
        self.n_spectral_features = 28000
        self.n_Cfeatures = 40
        self.c_embed = nn.Embedding(self.n_Cfeatures + 1, d_model, padding_idx=0)
        self.post_conv_transform = nn.Linear(128, d_model)

        #Kernel size = 5, Filters (out channels) = 64, in channels = 1
        self.conv1 = nn.Conv1d(1, 64, 5, stride = 1, padding = 'valid')
        #Max pool of size 12 with stride 12
        self.pool1 = nn.MaxPool1d(12)
        #Kernel size of 9, Filters (out channels) = 128, in channels = 64
        self.conv2 = nn.Conv1d(64, 128, 9, stride = 1, padding = 'valid')
        #Max pool of size 20 with stride 20
        self.pool2 = nn.MaxPool1d(20)
        self.relu = nn.ReLU()

    def _separate_spectra_components(self, x: Tensor):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)
        spectral_x = x[:, :, :self.n_spectral_features]
        cnmr_x = x[:, :, self.n_spectral_features:self.n_spectral_features + self.n_Cfeatures]
        mol_x = x[:, :, self.n_spectral_features + self.n_Cfeatures:]
        return spectral_x, cnmr_x, mol_x
    
    def _embed_cnmr(self, cnmr: Tensor):
        """Embeds the binary tensor into a continuous space
        Convert to 0-indexed indices, pad with 40
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
        spectra = self.pool2(spectra) #(N, 128, 116)
        spectra = torch.transpose(spectra, 1, 2)
        return self.post_conv_transform(spectra)

    def forward(self, x):
        spectra, cnmr, mol = self._separate_spectra_components(x)   
        cnmr_embed = self._embed_cnmr(cnmr)
        spectra_embed = self._embed_spectra(spectra)
        #print(spectra_embed.shape)
        #print(cnmr_embed.shape)
        return torch.cat((spectra_embed, cnmr_embed), dim = 1)

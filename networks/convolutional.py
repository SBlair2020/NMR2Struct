
import torch
import numpy as np
from torch import nn, Tensor

class H1Embed(nn.Module):
    """Docstring"""
    def __init__(self):
        #Kernel size = 5, Filters (out channels) = 64, in channels = 1
        self.conv1 = nn.Conv1d(1, 64, 5, stride = 1, padding = 'valid')
        #Max pool of size 12 with stride 12
        self.pool1 = nn.MaxPool1d(12)
        #Kernel size of 9, Filters (out channels) = 128, in channels = 64
        self.conv2 = nn.Conv1d(64, 128, 9, stride = 1, padding = 'valid')
        #Max pool of size 20 with stride 20
        self.pool2 = nn.MaxPool1d(20)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(start_dim = 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #Linear layers
        self.linear1 = nn.Linear(14848, 256)

        self.pretranspose = nn.Sequential([self.conv1, self.relu, self.pool1, self.conv2, self.relu, self.pool2])
        self.posttranspose = nn.Sequential([self.flatten, self.dropout, self.linear1, self.relu])
        

    def forward(self, x):
        x = self.pretranspose(x)
        x = torch.transpose(x, 1, 2)
        return self.posttranspose(x)
        


class C13Embed(nn.Module):
    """ Linear + ReLU """
    def __init__(self, n_Cfeatures):
        self.linear = nn.Linear(n_Cfeatures, 36).squeeze(1)
    
    def forward(self, x):
        return nn.ReLU()(self.linear(x))


class NMRConvNet(nn.Module):

    '''
    A translation of the CNN model used for interpreting spectral and chemical inputs in the first paper. 
    When rewriting the model from Keras to PyTorch, be sure to transpose the data before flattening it since 
    there is a mismatch of variables between the Keras and PyTorch specifications of the 1D convoluational layers 
    (see https://discuss.pytorch.org/t/how-to-transform-conv1d-from-keras/13738/6)
    '''


    model_id = 'CNN'
    
    def __init__(self, n_spectral_features, n_Cfeatures, n_molfeatures, n_substructures, concat = True,
                 dtype = torch.float):
        '''
        n_spectral_features: The number of spectral features, i.e. 28000
        n_Cfeatures: The number of CNMR features, i.e. 40
        n_molfeatures: The number of chemical formula features, i.e. 5
        n_substructures: The number of substructures to predict for. This is used for 
            constructing a single linear head for each substructure
        concat: Whether the output of the CNN is concatenated into a single output or 
            generated as a list of outputs. Defaults to True
        dtype: Model datatype. Default is torch.float
        '''
        super().__init__()
        self.n_Cfeatures = n_Cfeatures
        self.n_molfeatures = n_molfeatures
        self.n_spectral_features = n_spectral_features
        self.concat = concat
        self.dtype = dtype

        self.h1_embed = H1Embed()

        tot_num = 256
        if self.n_Cfeatures > 0:
            self.c13_embed = C13Embed(self.n_Cfeatures)
            tot_num += 36
        
        if self.n_molfeatures > 0:
            self.linearmol = nn.Linear(self.n_molfeatures, 8)
            tot_num += 8
        
        self.linear2 = nn.Linear(tot_num, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.substruct_pred_heads = nn.ModuleList([
            nn.Linear(1024, 1) for _ in range(n_substructures)
        ])

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (batch_size, 1, seq_len)
        '''
        # Separate out the features contained within the input vector
        spectral_x = x[:, :, :self.n_spectral_features]
        cnmr_x = x[:, :, self.n_spectral_features:self.n_spectral_features + self.n_Cfeatures]
        mol_x = x[:, :, self.n_spectral_features + self.n_Cfeatures:]

        spectral_x = self.h1_embed(spectral_x)

        # Mix in the information from the CNMR and chemical formula
        if self.n_Cfeatures > 0:
            cnmr_x = self.c13_embed(cnmr_x)
            spectral_x = torch.cat((spectral_x, cnmr_x), dim = -1)

        # Preserve the option to include features from the chemical formula
        if self.n_molfeatures > 0:
            mol_x = self.linearmol(mol_x).squeeze(1)
            mol_x = self.relu(mol_x)
            spectral_x = torch.cat((spectral_x, mol_x), dim = -1)

        spectral_x = self.linear2(spectral_x)
        spectral_x = self.relu(spectral_x)
        spectral_x = self.linear3(spectral_x)
        spectral_x = self.relu(spectral_x)

        # TODO: test if this is necessary
        spectral_x = [self.sigmoid(head(spectral_x)) for head in self.substruct_pred_heads]

        if self.concat:
            spectral_x = torch.cat(spectral_x, dim = -1)

        return spectral_x
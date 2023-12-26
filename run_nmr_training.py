import os
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from data_processing import sliced_cross_entropy


# view towards hydra 
# argparse for now
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('savepath', type = str)
args = parser.parse_args()

spectra_file = 'data/147k_spectra.h5'
labels_file = 'data/147k_labels.h5'
smiles_file = 'data/147k_smiles.h5'

NMR_dataset = NMRSpectraData(spectra_file, labels_file, smiles_file)

model = NMRConvNet()

# run training loop
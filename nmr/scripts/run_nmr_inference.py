import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import yaml
from nmr.data import create_dataset
from nmr.models import create_model
from nmr.training import create_optimizer, fit
import numpy as np
from typing import Optional, Union
import h5py
import random
import os
import pickle as pkl

def get_args() -> dict:
    raise NotImplementedError

def dtype_convert(dtype: str) -> torch.dtype:
    '''Maps string to torch dtype'''
    dtype_dict = {
        'float32': torch.float,
        'float64': torch.double,
        'float16': torch.half
    }
    return dtype_dict[dtype]

def main() -> None:
    raise NotImplementedError  

if __name__ == '__main__':
    main()
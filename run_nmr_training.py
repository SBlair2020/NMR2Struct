import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import yaml
from data import create_dataset
from models import create_model
from training import create_optimizer, fit
import numpy as np
from typing import Optional
import h5py

def get_args(yaml_path: str) -> dict:
    '''Parses the passed yaml file to get arguments'''
    return yaml.safe_load(open(yaml_path, 'r'))

def dtype_map(dtype: str) -> torch.dtype:
    '''Maps string to torch dtype'''
    dtype_dict = {
        'float32': torch.float,
        'float64': torch.double,
        'float16': torch.half
    }
    return dtype_dict[dtype]

def split_data_subsets(dataset: Dataset,
                       splits: Optional[str],
                       train_size: float = 0.8,
                       val_size: float = 0.1,
                       test_size: float = 0.1) -> tuple[Dataset, Dataset, Dataset]:
    '''Splits the dataset using indices from passed file
    Args:
        dataset: The dataset to split
        splits: The path to the numpy file with the indices for the splits
        train_size: The fraction of the dataset to use for training
        val_size: The fraction of the dataset to use for validation
        test_size: The fraction of the dataset to use for testing
    '''
    if splits is not None:
        print(f"Splitting data using indices from {splits}")
        split_indices = np.load(splits, allow_pickle = True)
        train, val, test = split_indices['train'], split_indices['val'], split_indices['test']
        return torch.utils.data.Subset(dataset, train), torch.utils.data.Subset(dataset, val), \
            torch.utils.data.Subset(dataset, test)
    else:
        print(f"Splitting data using {train_size} train, {val_size} val, {test_size} test")
        train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train, val, test

# view towards hydra 
# argparse for now
parser = argparse.ArgumentParser(description='Run NMR training')
parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
args = get_args(parser.parse_args())

# Separate arguments
global_args = args['global_args']
dataset_args = args['data']
model_args = args['model']
training_args = args['training']

# Set up consistent device, datatype, and seed (if applicable)
device = torch.device('cuda:0' if global_args['ngpus'] > 0 else 'cpu')
dtype = dtype_map[global_args['dtype']]
if global_args['seed'] is not None:
    torch.manual_seed(global_args['seed'])

# Set up dataset, model, optimizer, loss, and scheduler
dataset = create_dataset(dataset_args, dtype, device)
model = create_model(model_args, dtype, device)
model.to(dtype).to(device)
optimizer = create_optimizer(model, model_args, training_args, dtype, device)
loss_fn = getattr(nn, training_args['loss_fn'])
if training_args['loss_fn_args'] is not None:
    loss_fn = loss_fn(**training_args['loss_fn_args'])

if training_args['scheduler'] is not None:
    scheduler_raw = getattr(optim.lr_scheduler, training_args['scheduler'])
    scheduler = scheduler_raw(optimizer, **training_args['scheduler_args'])
else:
    scheduler = None

# Set up dataloaders
train_set, val_set, test_set = split_data_subsets(dataset, 
                                                  training_args['splits'],
                                                  training_args['train_size'],
                                                  training_args['val_size'],
                                                  training_args['test_size'])

train_loader = DataLoader(train_set, batch_size = training_args['batch_size'], **training_args['dloader_args'])
val_loader = DataLoader(val_set, batch_size = training_args['batch_size'], **training_args['dloader_args'])
test_loader = DataLoader(test_set, batch_size = training_args['batch_size'], **training_args['dloader_args'])

# Set up tensorboard writer
writer = SummaryWriter(log_dir = global_args['savedir'])

# Train
losses = fit(model=model,
             train_loader=train_loader,
             val_loader=val_loader,
             test_loader=test_loader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             nepochs=training_args['nepochs'],
             save_dir=global_args['savedir'],
             writer=writer,
             scheduler=scheduler,
             top_checkpoints_n=training_args['top_checkpoints_n'],
             loss_metric=training_args['checkpoint_loss_metric'],
             write_freq=training_args['write_freq'],
             test_freq=training_args['test_freq'],
             prev_epochs=training_args['prev_epochs']
             )

with h5py.File(f"{global_args['savedir']}/losses.h5", "w") as f:
    train_losses, val_losses, test_losses, model_names = losses
    f.create_dataset("train_losses", data = train_losses)
    f.create_dataset("val_losses", data = val_losses)
    f.create_dataset("test_losses", data = test_losses)
    f.create_dataset("model_names", data = model_names)
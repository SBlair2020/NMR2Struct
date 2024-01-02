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
from typing import Optional, Union
import h5py
import random
import os

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR training')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    args = parser.parse_args()
    listdoc =  yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['data'],
        listdoc['model'],
        listdoc['training']
    )

def seed_everything(seed: Union[int, None]) -> int:
    if seed is None:
        seed = random.randint(0, 100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dtype_convert(dtype: str) -> torch.dtype:
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
        assert(train_size + val_size + test_size == 1)
        print(f"Splitting data using {train_size} train, {val_size} val, {test_size} test")
        train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train, val, test

def main():
    # view towards hydra 
    # argparse for now
    # Separate arguments
    global_args, dataset_args, model_args, training_args = get_args()

    # Set up consistent device, datatype, and seed
    device = torch.device('cuda:0' if global_args['ngpus'] > 0 else 'cpu')
    dtype = dtype_convert(global_args['dtype'])
    _ = seed_everything(global_args['seed'])

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

    # Set up seeding in accordance with https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_set, worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
    val_loader = DataLoader(val_set, worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
    test_loader = DataLoader(test_set, worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])

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

if __name__ == '__main__':
    main()
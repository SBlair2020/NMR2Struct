import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from nmr.data import create_dataset
from nmr.models import create_model
from nmr.training import create_optimizer, fit
import nmr.training.loss_fxns as loss_fxns
import h5py
import pickle as pkl
from .top_level_utils import (
    seed_everything, 
    seed_worker, 
    dtype_convert, 
    save_completed_config, 
    split_data_subsets,
    save_train_history,
    save_token_size_dict,
    specific_update
)

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

def main() -> None:
    # view towards hydra 
    # argparse for now
    # Separate arguments
    print("Parsing arguments...")
    global_args, dataset_args, model_args, training_args = get_args()

    # Set up consistent device, datatype, and seed
    print("Setting up device, datatype, and seed...")
    device = torch.device('cuda:0' if global_args['ngpus'] > 0 else 'cpu')
    dtype = dtype_convert(global_args['dtype'])
    _ = seed_everything(global_args['seed'])

    print("Initializing dataset, model, optimizer, loss, and scheduler...")
    # Set up dataset, model, optimizer, loss, and scheduler
    dataset, updated_dataset_args = create_dataset(dataset_args, dtype, device)
    dataset.save_smiles_alphabet(global_args['savedir'])
    size_dict = dataset.get_sizes()
    token_dict = dataset.get_ctrl_tokens()
    total_dict = {**size_dict, **token_dict}
    #Fix target pad token as ignore index
    tgt_pad_token = total_dict['tgt_pad_token']
    total_dict['ignore_index'] = tgt_pad_token if tgt_pad_token is not None else -100
    print(total_dict)

    #Update model args
    model_args = specific_update(model_args, total_dict)
    #Update training args
    training_args = specific_update(training_args, total_dict)

    print(f"Max sequence lengths: {dataset.get_max_seq_len()}")

    model, updated_model_args = create_model(model_args, dtype, device)
    model.to(dtype).to(device)

    print("Initializing model parameters with Xavier uniform")
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    print("Total number of trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = create_optimizer(model, model_args, training_args, dtype, device)
    loss_fn = getattr(loss_fxns, training_args['loss_fn'])

    if training_args['loss_fn_args'] is not None:
        loss_fn = loss_fn(**training_args['loss_fn_args'])
    else:
        loss_fn = loss_fn()
    
    if hasattr(loss_fn, 'ignore_index'):
        print(f"Setting ignore index for loss function to {loss_fn.ignore_index}")

    if training_args['scheduler'] is not None:
        scheduler_raw = getattr(optim.lr_scheduler, training_args['scheduler'])
        scheduler = scheduler_raw(optimizer, **training_args['scheduler_args'])
    else:
        scheduler = None

    # Set up dataloaders
    print("Setting up dataloaders...")
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

    # Save completed config
    tot_config = {
        'global_args' : global_args,
        'data' : updated_dataset_args,
        'model' : updated_model_args,
        'training' : training_args
    }
    save_completed_config('full_train_config.yaml', tot_config, global_args['savedir'])

    # Train
    print("Beginning training")
    losses = fit(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                test_dataloader=test_loader,
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
    
    save_train_history(global_args['savedir'], losses)
    save_token_size_dict(global_args['savedir'], total_dict)

if __name__ == '__main__':
    main()
import numpy as np
import pickle, os, shutil
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter

def train_loop(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: Callable[[Tensor, Tensor], Tensor], 
               optimizer: torch.optim.Optimizer, 
               epoch: int, writer: torch.utils.tensorboard.SummaryWriter, 
               scheduler: Optional[torch.optim.lr_scheduler.LambdaLR], 
               write_freq: int = 100):
    """Model training loop
    Args:
        model: The model to train
        dataloader: The dataloader for the training dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        optimizer: The optimizer for training the model
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        scheduler: The optional learning rate scheduler
        write_freq: The frequency for printing loss information
    """
    tot_loss = 0
    model.train()
    for ibatch, (x, y) in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        loss = model.get_loss(x,y,loss_fn) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Step the learning rate scheduler too based on the current optimizer step
        if scheduler is not None:
            scheduler.step()
        if (ibatch % write_freq == 0):
            print(f"Epoch: {epoch}\tBatch:{ibatch}\tTrain Loss:{loss.item()}")
        tot_loss += loss.item()
        writer.add_scalar("Training Step Loss", loss.item(), inner_step)
    writer.add_scalar("Avg. Epoch Train Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)

def validation_loop(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss_fn: Callable[[Tensor, Tensor], Tensor], 
                    epoch: int, writer: torch.utils.tensorboard.SummaryWriter, 
                    write_freq: int = 100):
    """Model validation loop
    Args:
        model: The model to validate
        dataloader: The dataloader for the validation dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        write_freq: The frequency for printing loss information
    """
    tot_loss = 0
    model.eval()
    with torch.no_grad():
        for ibatch, (x, y) in enumerate(dataloader):
            inner_step = int(( epoch * len(dataloader)) + ibatch)
            loss = model.get_loss(x,y,loss_fn) 
            if (ibatch % write_freq == 0):
                print(f"Epoch: {epoch}\tBatch:{ibatch}\tValidation Loss:{loss.item()}")
            tot_loss += loss.item()
            writer.add_scalar("Validation Step Loss", loss.item(), inner_step)
        writer.add_scalar("Avg. Epoch Validation Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)

def test_loop(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: Callable[[Tensor, Tensor], Tensor], 
                epoch: int, writer: torch.utils.tensorboard.SummaryWriter, 
                write_freq: int = 100):
    """Model test loop
    Args:
        model: The model to test
        dataloader: The dataloader for the test dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        write_freq: The frequency for printing loss information
    """
    tot_loss = 0
    model.eval()
    with torch.no_grad():
        for ibatch, (x, y) in enumerate(dataloader):
            inner_step = int(( epoch * len(dataloader)) + ibatch)
            loss = model.get_loss(x,y,loss_fn) 
            if (ibatch % write_freq == 0):
                print(f"Epoch: {epoch}\tBatch:{ibatch}\Test Loss:{loss.item()}")
            tot_loss += loss.item()
            writer.add_scalar("Test Step Loss", loss.item(), inner_step)
        writer.add_scalar("Avg. Epoch Test Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)
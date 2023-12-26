import numpy as np
import pickle, os, shutil
import torch
import torch.nn as nn
#Project structural imports
from trainer import NMRML
from Models import NMRCNNModel, subs_weighted_BCE
from data import SpectraHDF5Dataset

# from nmr.networks import NMRConvNet

def train_loop(get_loss, model, loss, optimizer, scheduler, epoch, write_freq=100):
    tot_loss = 0
    for ibatch, (x, y) in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        loss = get_loss(x,y,model,loss) # TODO make this a class method
        optimizer.zero_grad()
        loss.backward()
        #self.writer.add_scalar("Step Learning Rate", self.optimizer.param_groups[0]['lr'], inner_step)
        optimizer.step()
        #Step the learning rate scheduler too based on the current optimizer step
        if scheduler is not None:
            scheduler.step()
        if (ibatch % 100 == 99):
            print(f"Epoch: {epoch}\tBatch:{ibatch}\tTrain Loss:{loss.item()}")
        tot_loss += loss.item()
        #self.writer.add_scalar("Training Step Loss", loss.item(), inner_step)
    #self.writer.add_scalar("Avg. Epoch Train Loss", tot_loss / len(self.train_loader), epoch)
    return tot_loss / len(dataloader)

def validation_loop():
    return NotImplementedError

def test_loop():
    return NotImplementedError
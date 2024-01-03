from nmr.data.dataset_base import NMRDataset
import torch

def create_dataset(dataset_args: dict, dtype: torch.dtype, device: torch.device) -> NMRDataset:
    return NMRDataset(
        dtype = dtype, 
        device = device, 
        **dataset_args
    ), dataset_args
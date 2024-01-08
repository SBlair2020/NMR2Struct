import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from nmr.data import create_dataset
from nmr.models import create_model
from nmr.inference.inference import run_inference
from .top_level_utils import (
    seed_everything,
    seed_worker,
    dtype_convert,
    split_data_subsets,
    save_inference_predictions,
    save_token_size_dict
)
from accelerate import Accelerator
from typing import Any

#Necessary functions for distributed data parallel inference

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR inference')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    args = parser.parse_args()
    listdoc =  yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['data'],
        listdoc['model'],
        listdoc['inference']
    )

def specific_update(mapping: dict[str, Any], update_map: dict[str, Any]) -> dict[str, Any]:
    """Recursively update keys in a mapping with the values specified in update_map"""
    mapping = mapping.copy()
    for k, v in mapping.items():
        #Give precedence to existing parameter settings
        if (k in update_map) and (not isinstance(v, dict)) and (v is None):
            mapping[k] = update_map[k]
        elif isinstance(v, dict):
            mapping[k] = specific_update(v, update_map)
    return mapping

def main() -> None:
    accelerator = Accelerator()
    _ = seed_everything(global_args['seed'])

    print("Parsing arguments...")
    global_args, dataset_args, model_args, inference_args = get_args()

    dtype = dtype_convert(global_args['dtype'])
    device = accelerator.device

    dataset, updated_dataset_args = create_dataset(dataset_args, dtype, device)
    size_dict = dataset.get_sizes()
    token_dict = dataset.get_ctrl_tokens()
    total_dict = {**size_dict, **token_dict}
    inference_args = specific_update(inference_args, total_dict)

    model, updated_model_args = create_model(model_args, dtype, device, addn_opts = total_dict)
    #At this point, the correct model checkpoint has been loaded
    model.to(dtype).to(device)

    #Set up dataloaders
    train_set, val_set, test_set = split_data_subsets(dataset, 
                                                    inference_args['splits'],
                                                    inference_args['train_size'],
                                                    inference_args['val_size'],
                                                    inference_args['test_size'])
    
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_set, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              **inference_args['dloader_args'])
    val_loader = DataLoader(val_set, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              **inference_args['dloader_args'])
    test_loader = DataLoader(test_set, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              **inference_args['dloader_args'])
    
    #Prepare for distributed inference using accelerator
    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader
    )

    #Run inference
    print("Running inference...")
    if ('train' in inference_args['sets_to_run']):
        train_predictions = run_inference(model,
                                        train_loader,
                                        device=device 
                                        **inference_args['run_inference_args']
                                        )
    else:
        train_predictions = None
    
    if ('val' in inference_args['sets_to_run']):
        val_predictions = run_inference(model,
                                        val_loader,
                                        device=device, 
                                        **inference_args['run_inference_args']
                                        )
    else:
        val_predictions = None
        
    if ('test' in inference_args['sets_to_run']):
        test_predictions = run_inference(model,
                                        test_loader,
                                        device=device, 
                                        **inference_args['run_inference_args']
                                        )
    else:
        test_predictions = None
        
    save_inference_predictions(global_args['savedir'], 
                               train_predictions,
                               val_predictions,
                               test_predictions)
    save_token_size_dict(global_args['savedir'], total_dict)

if __name__ == '__main__':
    main()
import os
from multiprocessing import Pool
from nmr.analysis.metric_fxns import (
    compute_molecule_BCE,
    compute_total_substruct_metrics
)
from typing import Callable
import h5py
import numpy as np
from nmr.analysis.util import (
    sanitize_prediction_set,
    construct_substructure_mols
)

#TODO: Add specific methods for SMILES and other analyses

def run_process_parallel(f: Callable, 
                         f_addn_args: dict,
                         pred_sets: list[h5py.File], 
                         num_processes: int):
    """Wraps an analysis function to run in parallel.

    Args:
        f: The inference analysis function that should at least take predictions and targets as arguments.
        f_addn_args: Additional arguments to pass to f.
        pred_sets: The list of h5py.File objects containing the predictions and targets. 
        num_processes: The number of processes to use for multiprocessing.

    Notes: 
        This method iterates over every h5py.File pointer contained in pred_sets and assumes that each 
        h5py.File pointer contains only the 'targets' and 'predictions'. This is done in lieu of merging
        all the data into a single file. 
    """
    assert(len(pred_sets) > 0)
    all_results = []
    for pset in pred_sets:
        assert('targets' in pset.keys())
        assert('predictions' in pset.keys())
        #Divide by the number of processes
        target_chunks = np.array_split(pset['targets'], num_processes)
        pred_chunks = np.array_split(pset['predictions'], num_processes)

        pool = Pool(processes = num_processes)
        results = []
        for i in range(num_processes):
            results.append(pool.apply_async(f, args = (pred_chunks[i], target_chunks[i]), kwds = f_addn_args))
        
        pool.close()
        pool.join()

        results = [r.get() for r in results]
        all_results.extend(results)
    return all_results

def process_SMILES_predictions(predictions: np.ndarray,
                               targets: np.ndarray,
                               substructures: str = None) -> tuple[
                                   list[str], list[list[tuple[str, float]]],
                                   list[str], list[list[str]]
                               ]:
    '''Processes SMILES predictions generated by the model

    Args: 
        predictions: The sets of unsanitized predictions generated by the model
        targets: The target SMILES strings
        substructures: Path to file of SMARTS strings representing the set of substructures

    For paralle processing, the substructures kwarg should be provided
    '''
    #Sanitize to obtain the good + bad targets and predictions
    (good_targets, good_predictions), (bad_targets, bad_predictions) = sanitize_prediction_set(predictions,
                                                                                               targets)
    substructure_strings = np.load(substructures, allow_pickle = True)
    substruct_mols = construct_substructure_mols(substructure_strings)
    good_targets, preds_with_losses = compute_molecule_BCE(good_predictions,
                                                      good_targets,
                                                      substruct_mols)
    return good_targets, preds_with_losses, bad_targets, bad_predictions

def process_substructure_predictions(predictions: np.ndarray,
                                     targets: np.ndarray) -> dict:
    """Processes substructure predictions by the model
    Args:

        predictions: The collated predictions from all the prediction sets
        targets: The collated targets from all the prediction sets
    """
    return compute_total_substruct_metrics(predictions, targets)
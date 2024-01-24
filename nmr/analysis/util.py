from rdkit import Chem as chem
import numpy as np
import warnings
import os, re
import h5py
from typing import Callable

### Methods for sanitizing SMILES strings ###
def count_num_heavy(smi: str) -> int:
    """Counts the number of heavy (non-hydrogen) atoms in a SMILES string"""
    mol = chem.MolFromSmiles(smi)
    total = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            total += 1
    return total

def construct_substructure_mols(substructures: list[str]) -> list[chem.Mol]:
    """
    Converts an array of substructures to an array of 
    molecules 
    """
    subs_mol = [chem.MolFromSmarts(s) for s in substructures]
    for s in subs_mol:
        s.UpdatePropertyCache()
    return subs_mol

def mols_to_labels(mols: list[chem.Mol], 
                   subs_mol: list[chem.Mol]) -> np.ndarray:
    """
    subs_mol
    out: labels
    """
    labels = []
    for mol in mols:
        labels.append([mol.HasSubstructMatch(s) for s in subs_mol])
    return np.array(labels)

def sanitize_smiles(smi: str) -> chem.Mol | None:
    """
    Returns None if the SMILES string is invalid
    """
    try: 
        mol = chem.MolFromSmiles(smi)
        mol_cpy = mol.__copy__()
        chem.SanitizeMol(mol_cpy)
        return mol_cpy
    except:
        warnings.warn("Invalid SMILES string detected after first conversion attempt")
        return None
    
def sanitize_single_pred(target: str,
                         predictions: np.ndarray) -> tuple[tuple[str, list[chem.Mol]] | None, bool]:
    """Performs sanitization of SMILES strings and returns a tuple of the target and valid molecules or None if the SMILES string is invalid
    Args:
    targets: set of target smiles strings
    predictions: set of predicted smiles strings

    Note: The strings are all expected to be byte strings loaded from an h5 file,
        so they must be decoded using utf-8 first. The function returns the original
        predictions if no valid predictions are found.
    """
    target = target.decode('utf-8')
    predictions = [p.decode('utf-8') for p in predictions]
    valid_predictions = []
    for pred in predictions:
        sanitize_result = sanitize_smiles(pred)
        if sanitize_result is not None:
            valid_predictions.append(sanitize_result)
    if len(valid_predictions) == 0:
        return target, predictions, False
    else:
        return target, valid_predictions, True

def sanitize_prediction_set(predictions: np.ndarray,
                            targets: np.ndarray) -> tuple[
                                tuple[list[str], list[list[chem.Mol]]],
                                tuple[list[str], list[list[str]]]
                                ]:
    """
    Sanitizes a set of predictions

    predictions: 2D array of all predicted smiles strings
    targets: 1D array of all target smiles strings

    Returns good predictions and bad predictions

    Can be run in parallel via multiprocessing
    """
    assert(len(predictions) == len(targets))
    good_targets = []
    good_predictions = []
    bad_targets = []
    bad_predictions = []
    for i in range(len(targets)):
        curr_tgt, curr_pred = targets[i], predictions[i]
        curr_tgt, valid_predictions, processed = sanitize_single_pred(curr_tgt, curr_pred)
        if processed:
            good_targets.append(curr_tgt)
            good_predictions.append(valid_predictions)
        else:
            bad_targets.append(curr_tgt)
            bad_predictions.append(curr_pred)
    return (good_targets, good_predictions), (bad_targets, bad_predictions)

def intake_data(savedir: str, 
                pattern: str) -> list[np.ndarray]:
    '''Aggregates all the predictions from the savedir using the given pattern
    
    savedir: The directory where predictions are saved
    pattern: The pattern to match for the predictions as a regular expression string

    Opens a number of h5file pointers to all files in savedir that match pattern
    '''
    all_files = os.listdir(savedir)
    relevant_files = list(filter(lambda x : re.match(pattern, x), all_files))
    prediction_files = [h5py.File(os.path.join(savedir, f), "r") for f in relevant_files]
    return prediction_files

class PredictionView:
    def __init__(self,
                 curr_set: str,
                 file_handles: list[h5py.File]):
        
        self.file_handles = [f[curr_set] for f in file_handles]
        self.len_lower_bounds = np.array([len(f['targets']) for f in self.file_handles]).cumsum()
    
    def __getitem__(self, idx: int):
        if idx < 0: 
            raise IndexError("Negative indices not supported")
        elif idx >= self.len_lower_bounds[-1]:
            raise IndexError("Index out of bounds")
        else:
            file_idx = np.argmax(idx < self.len_lower_bounds)
            if file_idx == 0:
                return self.file_handles[file_idx][idx]
            else:
                return self.file_handles[file_idx][idx - self.len_lower_bounds[file_idx-1]]
            
### Inversion methods for converting back to binary substructure arrays ###
def convert_one_indexed_seq_to_binary(sequence: np.ndarray) -> np.ndarray:
    #Take care of 1-indexing during processing
    seq_shifted = sequence - 1 
    binary = np.zeros(957)
    binary[seq_shifted] = 1
    return binary

def apply_substruct_invert_fxn(predictions: np.ndarray,
                               inversion_fxn: Callable[[np.ndarray], np.ndarray],
                               padding_token: int = None) -> np.ndarray:
    '''Takes a sequence of potentially ragged predictions (padded with padding_token) and converts them
        back to a binary substructure representation
    
    Args:
        predictions: The set of predictions to convert back to binary
        inversion_fxn: The function to apply to each prediction to convert back to binary
        padding_token: The token used for padding the predictions
    '''
    print(f"Inverting {len(predictions)} predictions")
    inverted = []
    for i in range(len(predictions)):
        curr_pred = predictions[i]
        if padding_token is not None:
            curr_pred = curr_pred[curr_pred != padding_token]
        inverted.append(inversion_fxn(curr_pred))
    assert(len(inverted) == len(predictions))
    return np.array(inverted)

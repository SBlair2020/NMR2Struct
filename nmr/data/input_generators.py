import numpy as np
from .tokenizer import BasicSmilesTokenizer
from typing import Tuple

def look_ahead_spectra(spectra: np.ndarray, eps: float) -> int:
    """Determines the maximum number of peaks for padding"""
    max_len = 0
    for i in range(len(spectra)):
        num_nonzero = np.sum(spectra[i] > eps)
        max_len = max(max_len, num_nonzero)
    return max_len

def look_ahead_substructs(labels: np.ndarray) -> int:
    """Determines the maximum sequence length for padding"""
    max_len = 0
    for i in range(len(labels)):
        max_len = max(max_len, np.count_nonzero(labels[i]))
    return max_len

class SubstructureRepresentationOneIndexed:
    """Processes binary substructure array to 1-indexed values with 0 padding"""
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        """
        Args:
            spectra_file: Path to the HDF5 file with spectra
            smiles_file: Path to the HDF5 file with smiles
            label_file: Path to the HDF5 file with substructure labels
            input_generator: Function name that generates the model input
            target_generator: Function name that generates the model target
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        self.pad_token = 0
        self.max_len = look_ahead_substructs(labels)
        self.alphabet_size = labels.shape[1] + 1
    
    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Transforms the input binary substructure array into shifted and padded 1-indexed array
        Args:
            substructures: Binary substructure array
        
        Example:
            original vector:  [0, 1, 0, 1, 1, 0, 0]
            shifted + padded vector: [2, 4, 5, 0, 0, 0, 0]
        """
        indices = np.arange(len(substructures)) + 1
        indices = indices * substructures
        nonzero_entries = indices[indices != 0]
        padded = np.pad(nonzero_entries, 
                        (0, self.max_len - len(nonzero_entries)), 
                        'constant', 
                        constant_values = (self.pad_token,))
        return padded
    
    def get_size(self) -> int:
        '''Returns the size of the input alphabet'''
        return self.alphabet_size
    
    def get_ctrl_tokens(self) -> tuple[int, int, int]:
        '''Returns the stop, start, and pad tokens in that order (inputs typically only have pad tokens)'''
        return (None, None, self.pad_token)
    
class SubstructureRepresentationBinary:
    """Dummy class for binary substructure representation and interface consistency"""
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        """
        Args:
            spectra_file: Path to the HDF5 file with spectra
            smiles_file: Path to the HDF5 file with smiles
            label_file: Path to the HDF5 file with substructure labels
            input_generator: Function name that generates the model input
            target_generator: Function name that generates the model target
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        self.alphabet_size = 2

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Returns the substructure array"""
        return np.expand_dims(substructures, axis = -1)
    
    def get_size(self) -> int:
        '''Returns the size of the input alphabet'''
        return self.alphabet_size
    
    def get_ctrl_tokens(self) -> tuple[int, int, int]:
        '''Returns the stop, start, and pad tokens in that order (inputs typically only have pad tokens)'''
        return (None, None, None)

# TODO: Methods for variable processing of spectral data (raw tokenized, etc.)
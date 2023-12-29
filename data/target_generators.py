import numpy as np
from .tokenizer import BasicSmilesTokenizer
from typing import Tuple

def look_ahead_smiles(smiles: np.ndarray, tokenizer: BasicSmilesTokenizer) -> int:
    """Determines the maximum length of the smiles strings in numbers of tokens"""
    max_len = 0
    for i in range(len(smiles)):
        tokens = tokenizer.tokenize(smiles[i])
        max_len = max(max_len, len(tokens))
    return max_len

def look_ahead_substructs(labels: np.ndarray) -> int:
    """Determines the maximum sequence length for padding"""
    max_len = 0
    for i in range(len(labels)):
        max_len = max(max_len, np.count_nonzero(labels[i]))
    return max_len

class SmilesRepresentationTokenized:
    """Processes SMILES strings into tokenized arrays with padding"""
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
        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.tokenizer = tokenizer
        self.max_len = look_ahead_smiles(smiles, self.tokenizer)
        self.index_map = {char: i for i, char in enumerate(alphabet)}
        self.alphabet_size = len(alphabet) + 3

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the input smiles string into tokenized array with padding"""
        tokenized_smiles = self.tokenizer.tokenize(smiles)
        full_seq = [self.index_map[char] for char in tokenized_smiles] + [self.stop_token]
        shifted_seq = [self.start_token] + full_seq[:-1]
        assert(len(shifted_seq) == len(full_seq))
        shifted_seq = np.pad(shifted_seq,
                            (0, self.max_len - len(shifted_seq)),
                            'constant',
                            constant_values = (self.pad_token,))
        full_seq = np.pad(full_seq,
                            (0, self.max_len - len(full_seq)),
                            'constant',
                            constant_values = (self.pad_token,))
        return shifted_seq, full_seq
    
    def get_size(self) -> int:
        '''Returns the size of the target alphabet'''
        return self.alphabet_size
    
class SubstructureRepresentationBinary:
    """Processes binary substructure representation, augmented with start and stop tokens"""
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
        self.pad_token = None
        self.start_token = 3
        self.stop_token = 2
        self.max_len = look_ahead_substructs(labels)
        self.alphabet_size = 4

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the input binary substructure array into tuple of arrays with stop and start tokens"""
        full_seq = np.concatenate((
            substructures,
            [self.stop_token]
        ))
        shifted_seq = np.concatenate((
            [self.start_token],
            full_seq[:-1]
        ))
        assert(shifted_seq.shape == full_seq.shape)
        return shifted_seq, full_seq
    
    def get_size(self) -> int:
        '''Returns the size of the target alphabet'''
        return self.alphabet_size
    
class SubstructureRepresentationOneIndexed:
    """Processes binary substructures into 1-indexed arrays with padding and ctrl tokens"""
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
        self.stop_token = labels.shape[1] + 1
        self.start_token = self.stop_token + 1
        self.max_len = look_ahead_substructs(labels)
        self.alphabet_size = labels.shape[1] + 3
    
    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the input binary substructure array into shifted and padded 1-indexed array"""
        full_seq = np.concatenate((
            substructures, 
            [1]
        ))
        indices = np.arange(len(full_seq)) + 1  
        indices = indices * full_seq
        full_seq = indices[indices != 0]
        shifted_seq = np.concatenate((
            [self.start_token],
            full_seq[:-1]
        ))
        assert(shifted_seq.shape == full_seq.shape)
        shifted_seq = np.pad(shifted_seq, 
                            (0, self.max_len - len(shifted_seq)), 
                            'constant', 
                            constant_values = (self.pad_token,))
        full_seq = np.pad(full_seq, 
                            (0, self.max_len - len(full_seq)),
                            'constant',
                            constant_values = (self.pad_token,))
        return shifted_seq, full_seq
    
    def get_size(self) -> int:
        '''Returns the size of the target alphabet'''
        return self.alphabet_size
from torch.utils.data import Dataset
import numpy as np
import h5py
from typing import List, Tuple
from nmr.data.tokenizer import BasicSmilesTokenizer
from functools import reduce
from rdkit import Chem as chem
import nmr.data.input_generators as input_generators
import nmr.data.target_generators as target_generators
import torch

class NMRDataset(Dataset):
    """Sketch of potential dataset class"""
    def __init__(self, 
                 spectra_file: str,
                 smiles_file: str,
                 label_file: str, 
                 input_generator: str,
                 target_generator: str, 
                 alphabet: str,
                 eps: float = 0.005,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
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
        self.spectra_h5 = h5py.File(spectra_file, 'r')
        self.label_h5 = h5py.File(label_file, 'r')

        self.spectra = self.spectra_h5['spectra']
        self.labels = self.label_h5['substructure_labels']
        self.smiles = np.load(smiles_file, allow_pickle = True)
        self.tokenizer = BasicSmilesTokenizer()
        self.eps = eps

        self.device = device
        self.dtype = dtype

        if alphabet is not None:
            self.alphabet = np.load(alphabet, allow_pickle = True)
        else:
            self.alphabet = self._determine_smiles_alphabet(self.smiles)

        #Initialize the input and target generators with a quick "look ahead" to 
        #   determine global information, i.e. max lengths and padding
        self.input_generator = getattr(input_generators, input_generator)(
            self.spectra, self.labels, self.smiles, 
            self.tokenizer, self.alphabet, self.eps
        )
        self.target_generator = getattr(target_generators, target_generator)(
            self.spectra, self.labels, self.smiles, 
            self.tokenizer, self.alphabet, self.eps
        )

    def __len__(self):
        return len(self.labels)

    def _determine_smiles_alphabet(self, smiles: List[str]):
        """Generates the alphabet from the set of smiles strings
        Args:
            smiles: list of smiles strings
        """
        token_sets = [set(self.tokenizer.tokenize(smi)) for smi in smiles]
        final_set = list(reduce(lambda x, y : x.union(y), token_sets))
        alphabet = sorted(final_set)
        return alphabet
    
    def save_smiles_alphabet(self, save_dir: str) -> None:
        '''Quickly saves the alphabet as a npy file'''
        with open(f"{save_dir}/alphabet.npy", "wb") as f:
            np.save(f, self.alphabet)
    
    def __getitem__(self, idx):
        spectra_data = self.spectra[idx]
        smiles_data = chem.CanonSmiles(self.smiles[idx].decode("utf-8"))
        label_data = self.labels[idx]
        model_input = self.input_generator.transform(spectra_data, smiles_data, label_data)
        model_target = self.target_generator.transform(spectra_data, smiles_data, label_data)
        model_input = torch.from_numpy(model_input).to(self.dtype).to(self.device)
        model_target = torch.from_numpy(model_target).to(self.dtype).to(self.device)
        return (model_input, smiles_data), model_target
    
    def get_sizes(self) -> dict[int, int]:
        """Returns the padding tokens for the input and target"""
        input_size = self.input_generator.get_size()
        target_size = self.target_generator.get_size()
        return {'source_size' : input_size, 'target_size' : target_size}
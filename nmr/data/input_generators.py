import numpy as np
from .tokenizer import BasicSmilesTokenizer
from scipy.signal import find_peaks
import warnings

def look_ahead_substructs(labels: np.ndarray) -> int:
    """Determines the maximum sequence length for padding"""
    max_len = 0
    for i in range(len(labels)):
        max_len = max(max_len, np.count_nonzero(labels[i]))
    return max_len

### Spectrum processing methods ###
def threshold_spectra(spectra: np.ndarray, eps: float) -> np.ndarray:
    """Sets values lower than eps to 0"""
    spectra[spectra < eps] = 0
    return spectra

def spectrum_extraction(spectrum: np.ndarray, criterion: str) -> np.ndarray:
    """Extracts the indices from a spectrum based on the given criterion"""
    if criterion == 'all_nonzero':
        indices = np.where(spectrum > 0)[0]
    elif criterion == 'find_peaks':
        indices, _ = find_peaks(spectrum)
    elif criterion == 'binary':
        indices = np.where(spectrum == 1)[0]
    else:
        raise ValueError("Invalid criterion for spectrum extraction")
    return indices

def select_points(spectra: np.ndarray, hnmr_criterion: str, cnmr_criterion: str) -> np.ndarray:
    hnmr_spectrum = spectra[:28000]
    cnmr_spectrum = spectra[28000:28040]
    hnmr_indices = spectrum_extraction(hnmr_spectrum, hnmr_criterion)   
    cnmr_indices = spectrum_extraction(cnmr_spectrum, cnmr_criterion)
    return hnmr_spectrum, cnmr_spectrum, hnmr_indices, cnmr_indices

def point_representation(representation_name: str,
                         hnmr_spectrum: np.ndarray,
                         cnmr_spectrum: np.ndarray,
                         hnmr_indices: np.ndarray,
                         cnmr_indices: np.ndarray,
                         hnmr_shifts: np.ndarray = None,
                         cnmr_shifts: np.ndarray = None,
                         bins: np.ndarray = None) -> np.ndarray:
    """Transforms the spectrum into the desired representation for downstream tasks
    Args:
        representation_name: The name of the representation to use, currently the following are implemented:
            'tokenized_indices': Returns the toeknized intensity values at the given indices and the indices themselves for 
                specific positional encodings
            'continuous_pair': Returns pairs of (x, y) for the selected points where x is the ppm shift and y is the 
                intensity value
        hnmr_spectrum: Numpy array of the HNMR intensities
        cnmr_spectrum: Numpy array of the CNMR intensities
        hnmr_indices: Numpy array of selected HNMR indices
        cnmr_indices: Numpy array of selected CNMR indices
        hnmr_shifts: Array of HNMR shifts. Required for 'continuous_pair' representation
        cnmr_shifts: Array of CNMR shifts. Required for 'continuous_pair' representation
        bins: np.ndarray, bin array to use for digitizing spectra
    """
    if representation_name == 'tokenized_indices':
        assert(bins is not None)
        all_indices = np.concatenate((hnmr_indices, cnmr_indices + 28000))
        all_intensities = np.concatenate((hnmr_spectrum[hnmr_indices], cnmr_spectrum[cnmr_indices]))
        tokenized_intensities = np.digitize(all_intensities, bins)
        return np.vstack((tokenized_intensities, all_indices))
    elif representation_name == 'continuous_pair':
        assert((hnmr_shifts is not None) and (cnmr_shifts is not None))
        selected_hnmr_shifts = hnmr_shifts[hnmr_indices]
        selected_hnmr_intensities = hnmr_spectrum[hnmr_indices]
        selected_cnmr_shifts = cnmr_shifts[cnmr_indices]
        selected_cnmr_intensities = cnmr_spectrum[cnmr_indices]
        hnmr_pairs = np.vstack((selected_hnmr_shifts, selected_hnmr_intensities)).T
        cnmr_pairs = np.vstack((selected_cnmr_shifts, selected_cnmr_intensities)).T
        return np.vstack((hnmr_pairs, cnmr_pairs))

def apply_padding(representation_name: str, 
                  processed_spectrum: np.ndarray,
                  padding_value: int,
                  max_len: int) -> np.ndarray:
    """Applies padding to the processed spectrum using the given padding value
    Args:
        representation_name: The name of the representation to use, consistent with point_representation()
        processed_spectrum: The processed spectrum to pad
        padding_value: The value to use for padding
        max_len: The maximum length to pad to
    """
    if representation_name == 'tokenized_indices':
        return np.pad(
            processed_spectrum,
            ((0, 0), (0, max_len - processed_spectrum.shape[1])),
            'constant',
            constant_values = (padding_value,)
        )
    elif representation_name == 'continuous_pair':
        return np.vstack((
            processed_spectrum, np.ones((max_len - processed_spectrum.shape[0], 2)) * padding_value
        ))

def look_ahead_spectra(spectra: np.ndarray, 
                       hnmr_criterion: str, 
                       cnmr_criterion: str,
                       eps: float) -> int:
    """Determines the maximum number of peaks for padding"""
    max_hnmr_len = -1
    max_cnmr_len = -1
    max_tot_len = -1
    for i in range(len(spectra)):
        _, _, hnmr_indices, cnmr_indices = select_points(threshold_spectra(spectra[i], eps), 
                                                         hnmr_criterion, 
                                                         cnmr_criterion)
        max_hnmr_len = max(max_hnmr_len, len(hnmr_indices)) 
        max_cnmr_len = max(max_cnmr_len, len(cnmr_indices))
        tot_len = len(hnmr_indices) + len(cnmr_indices)
        max_tot_len = max(max_tot_len, tot_len)
    #Keep these maximums separate for greater flexibility
    return max_hnmr_len, max_cnmr_len, max_tot_len

#Abstract base class for input generators, to be inherited by others
class InputGeneratorBase:
    #Getters have concrete implementations, but constructor and transform are not implemented
    def __init__(self,
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        pass

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        pass

    def get_size(self) -> int:
        return self.alphabet_size
    
    def get_ctrl_tokens(self) -> tuple[int, int, int]:
        return (self.stop_token, self.start_token, self.pad_token)
    
    def get_max_seq_len(self) -> int:
        return self.max_len

class SubstructureRepresentationOneIndexed(InputGeneratorBase):
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
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        self.pad_token = 0
        self.stop_token = None
        self.start_token = None
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
    
class SubstructureRepresentationBinary(InputGeneratorBase):
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
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        #We set the alphabet to 957 because binary representation requires 957 tokens 
        #   (even though each token is a 0 or 1)
        self.alphabet_size = 957
        self.pad_token = None
        self.stop_token = None
        self.start_token = None
        self.max_len = 957

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Returns the substructure array"""
        #We expand here because for substructure to structure, the transformer expects a 3D input
        return np.expand_dims(substructures, axis = -1)

class SpectrumRepresentationUnprocessed(InputGeneratorBase):

    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        
        self.pad_token = None
        self.stop_token = None
        self.start_token = None
        self.alphabet_size = 28045
        self.max_len = 28040
    
    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Returns the spectra array"""
        return spectra

class SpectrumRepresentationThresholdTokenized(InputGeneratorBase):
    """Selects peaks from the spectrum after thresholding and tokenizes them"""
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float,
                 hnmr_selection: str = 'all_nonzero',
                 cnmr_selection: str = 'all_nonzero',
                 nbins: int = 200):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
            hnmr_selection: The criterion to use for selecting HNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            cnmr_selection: The criterion to use for selecting CNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            nbins: The number of bins to use for digitizing the spectra

        Note: The additional arguments:
            hnmr_selection,
            cnmr_selection,
            nbins
        should be specified in the config file as additional arguments for the input generator

        Spectra are expected as arrays of normalized intensity with values in [0, 1]
        """
        
        self.hnmr_criterion = hnmr_selection
        self.cnmr_criterion = cnmr_selection
        self.eps = eps
        self.max_hnmr_len, self.max_cnmr_len, self.max_len = look_ahead_spectra(spectra, self.hnmr_criterion, self.cnmr_criterion, self.eps)
        self.pad_token = 0 
        self.stop_token = None
        self.start_token = None
        self.alphabet_size = nbins + 1
        self.bins = np.linspace(eps, 1, nbins)
        self.representation_name = 'tokenized_indices'

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        spectra = threshold_spectra(spectra, self.eps)
        hnmr_spectrum, cnmr_spectrum, hnmr_indices, cnmr_indices = select_points(spectra, self.hnmr_criterion, self.cnmr_criterion)
        processed_spectrum = point_representation(self.representation_name,
                                                  hnmr_spectrum,
                                                  cnmr_spectrum,
                                                  hnmr_indices,
                                                  cnmr_indices,
                                                  bins=self.bins)
        processed_spectrum = apply_padding(self.representation_name, 
                                           processed_spectrum, 
                                           self.pad_token, 
                                           self.max_len)  
        return processed_spectrum
    
class SpectrumRepresentationThresholdPairs(InputGeneratorBase):
    """Selects ALL non-zero peaks from the spectrum after thresholding and represents them as pairs"""
    def __init__(self, spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float,
                 hnmr_selection: str = 'all_nonzero',
                 cnmr_selection: str = 'all_nonzero',
                 hnmr_shifts: str = None,
                 cnmr_shifts: str = None):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
            hnmr_selection: The criterion to use for selecting HNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            cnmr_selection: The criterion to use for selecting CNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            hnmr_shifts: Path to a file specifying the HNMR shift values
            cnmr_shifts: Path to a file specifying the CNMR shift values

        Note: The additional arguments:
            hnmr_selection,
            cnmr_selection,
            hnmr_shifts,
            cnmr_shifts
        should be specified in the config file as additional arguments for the input generator

        Spectra are expected as arrays of normalized intensity with values in [0, 1]
        """
        #Handle shift initialization
        if hnmr_shifts is not None:
            self.hnmr_shifts = np.load(hnmr_shifts, allow_pickle=True)
        else:
            warnings.warn("No HNMR shifts provided, using default values from -2 to 12 ppm")
            self.hnmr_shifts = np.arange(-2, 12, 0.0005)
        if cnmr_shifts is not None:
            self.cnmr_shifts = np.load(cnmr_shifts, allow_pickle=True)
        else:
            warnings.warn("No CNMR shifts provided, using default values from -2 to 231 ppm")
            self.cnmr_shifts = np.linspace(-2, 231, 40)
        
        self.hnmr_criterion = hnmr_selection
        self.cnmr_criterion = cnmr_selection
        self.eps = eps
        self.max_hnmr_len, self.max_cnmr_len, self.max_len = look_ahead_spectra(spectra, self.hnmr_criterion, self.cnmr_criterion, self.eps)
        self.pad_token = -1000 
        self.stop_token = None
        self.start_token = None
        #Not relevant for this representation
        self.alphabet_size = self.max_len
        self.representation_name = 'continuous_pair'

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        spectra = threshold_spectra(spectra, self.eps)
        hnmr_spectrum, cnmr_spectrum, hnmr_indices, cnmr_indices = select_points(spectra, self.hnmr_criterion, self.cnmr_criterion)
        processed_spectrum = point_representation(self.representation_name,
                                                  hnmr_spectrum,
                                                  cnmr_spectrum,
                                                  hnmr_indices,
                                                  cnmr_indices,
                                                  hnmr_shifts=self.hnmr_shifts,
                                                  cnmr_shifts=self.cnmr_shifts)
        processed_spectrum = apply_padding(self.representation_name,
                                           processed_spectrum,
                                           self.pad_token,
                                           self.max_len)
        return processed_spectrum
    
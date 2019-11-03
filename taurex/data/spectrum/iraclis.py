from .array import ArraySpectrum
import numpy as np
import pickle

class IraclisSpectrum(ArraySpectrum):
    """
    Loads an observed spectrum from a text file and computes bin
    edges and bin widths. Spectrum must be 3-4 columns with ordering:
        1. wavelength
        2. spectral data
        3. error
        4. (optional) bin width
    
    If no bin width is present then they are computed.

    Parameters
    -----------
    filename: string
        Path to observed spectrum file. 

    """

    def __init__(self, filename):
        self._filename = filename
        try:
            with open(filename, 'rb') as f:
                database = pickle.load(f)
        except UnicodeDecodeError:
            with open(filename, 'rb') as f:
                database = pickle.load(f,encoding='latin1')       
        wl = database['spectrum']['wavelength']
        td = database['spectrum']['depth']
        err = database['spectrum']['error']
        width = database['spectrum']['width']

        final_array = np.vstack((wl, td, err, width)).T

        super().__init__(final_array)
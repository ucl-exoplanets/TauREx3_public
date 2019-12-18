from .array import ArraySpectrum
import numpy as np
from taurex.util.util import wnwidth_to_wlwidth


class TaurexSpectrum(ArraySpectrum):
    """
    Observation is a taurex spectrum from a HDF5 file

    An instrument function must have been used for this to work

    Parameters
    -----------
    filename: string
        Path to taurex spectrum HDF5 output.

    """

    def __init__(self, filename):
        super().__init__(self._load_from_hdf5(filename))

    def _load_from_hdf5(self, filename):
        import h5py

        with h5py.File(filename, 'r') as f:
            try:
                wngrid = f['Output']['Spectra']['instrument_wngrid'][:]
            except KeyError:
                self.error('Could not find instrument outputs in HDF5, '
                           'this was caused either by the HDF5 being a '
                           'retrieval output or not running with some '
                           'form of instrument in the forward model'
                           ' input par file')
                raise KeyError('Instrument output not found')

            spectrum = f['Output']['Spectra']['instrument_spectrum'][:]
            noise = f['Output']['Spectra']['instrument_noise'][:]
            wnwidth = f['Output']['Spectra']['instrument_wnwidth'][:]

        wlgrid = 10000/wngrid

        wlwidth = wnwidth_to_wlwidth(wngrid, wnwidth)

        return np.vstack((wlgrid, spectrum, noise, wlwidth)).T

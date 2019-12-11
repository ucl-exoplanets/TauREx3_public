from .spectrum import BaseSpectrum
import numpy as np
from taurex.util.util import wnwidth_to_wlwidth


class ArraySpectrum(BaseSpectrum):

    """
    Loads an observed spectrum from an array and computes bin
    edges and bin widths. Spectrum shape(nbins, 3-4) with 3-4 columns with
    ordering:
        1. wavelength (um)
        2. spectral data
        3. error
        4. (optional) bin width

    If no bin width is present then they are computed.

    Parameters
    -----------
    filename: string
        Path to observed spectrum file.

    """

    def __init__(self, spectrum):
        super().__init__(self.__class__.__name__)

        self._obs_spectrum = spectrum
        self._bin_widths = None
        self._bin_edges = None

        self._sort_spectrum()
        self._process_spectrum()

        self._wnwidths = wnwidth_to_wlwidth(self.wavelengthGrid,
                                            self._bin_widths)

    def _sort_spectrum(self):
        self._obs_spectrum = \
            self._obs_spectrum[self._obs_spectrum[:, 0].argsort(axis=0)[::-1]]

    def _process_spectrum(self):
        """
        Seperates out the observed data, error, grid and binwidths
        from array. If bin widths are not present then they are
        calculated here
        """
        if self.rawData.shape[1] == 4:
            self._bin_widths = self._obs_spectrum[:, 3]
            obs_wl = self.wavelengthGrid[::-1]
            obs_bw = self._bin_widths[::-1]

            bin_edges = np.zeros(shape=(len(self._bin_widths)*2,))

            bin_edges[0::2] = obs_wl - obs_bw/2
            bin_edges[1::2] = obs_wl + obs_bw/2
            # bin_edges[-1] = obs_wl[-1]-obs_bw[-1]/2.

            self._bin_edges = bin_edges[::-1]
        else:
            self.manual_binning()

    @property
    def rawData(self):
        """Data read from file"""
        return self._obs_spectrum

    @property
    def spectrum(self):
        """The spectrum itself"""
        return self._obs_spectrum[:, 1]

    @property
    def wavelengthGrid(self):
        """Wavelength grid in microns"""
        return self.rawData[:, 0]

    @property
    def wavenumberGrid(self):
        """Wavenumber grid in cm-1"""
        return 10000/self.wavelengthGrid

    @property
    def binEdges(self):
        """ Bin edges"""
        return 10000/self._bin_edges

    @property
    def binWidths(self):
        """bin widths"""
        return self._wnwidths

    @property
    def errorBar(self):
        """ Error bars for the spectrum"""
        return self.rawData[:, 2]

    def manual_binning(self):
        from taurex.util.util import compute_bin_edges
        """
        Performs the calculation of bin edges when none are present
        """
        self._bin_edges, self._bin_widths = \
            compute_bin_edges(self.wavelengthGrid)

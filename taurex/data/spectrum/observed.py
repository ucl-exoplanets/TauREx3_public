from .array import ArraySpectrum
import numpy as np


class ObservedSpectrum(ArraySpectrum):
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

        super().__init__(np.loadtxt(self._filename))

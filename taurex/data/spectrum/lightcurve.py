from .spectrum import BaseSpectrum
import numpy as np


class ObservedLightCurve(BaseSpectrum):
    """Loads an observed lightcurve from a file and computes bin
    edges and bin widths

    Parameters
    -----------
    filename: string
        File name of observed spectrum. Spectrum must be 3-4 columns
        1st column: 
            wavelength
        2nd column:
            spectral data
        3rd column:
            error
        4th(optional):
            bin width

    """

    def __init__(self,filename):
        super().__init__('observed_lightcurve')

    

    @property
    def spectrum(self):
        raise NotImplementedError

    @property
    def rawData(self):
        raise NotImplementedError

    @property
    def wavelengthGrid(self):
        raise NotImplementedError
    
    @property
    def wavenumberGrid(self):
        return 10000/self.wavelengthGrid

    @property
    def binEdges(self):
        raise NotImplementedError
    
    @property
    def binWidths(self):
        raise NotImplementedError


    @property
    def errorBar(self):
        raise NotImplementedError
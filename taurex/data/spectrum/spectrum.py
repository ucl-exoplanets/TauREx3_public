
from taurex.log import Logger



class BaseSpectrum(Logger):
    """A base class where spectrums are loaded (or later created)"""


    def __init__(self,name):
        super().__init__(name)

    

    @property
    def spectrum(self):
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




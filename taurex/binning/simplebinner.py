from .binner import Binner
from taurex.util import bindown

class SimpleBinner(Binner):

    def __init__(self,wavenumber_grid):

        self._wngrid = wavenumber_grid

    
    def bindown(self,wngrid,spectrum):
        
        return bindown(wngrid,spectrum,self._wngrid)





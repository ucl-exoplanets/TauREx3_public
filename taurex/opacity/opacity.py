from taurex.log import Logger
import numpy as np

class Opacity(Logger):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,name):
        super().__init__(name)


    @property
    def resolution(self):
        raise NotImplementedError
    
    @property
    def moleculeName(self):
        raise NotImplementedError

    @property
    def wavenumberGrid(self):
        raise NotImplementedError

    @property
    def temperatureGrid(self):
        raise NotImplementedError
    
    @property
    def pressureGrid(self):
        raise NotImplementedError


    def compute_opacity(self,temperature,pressure):
        raise NotImplementedError

    def opacity(self,wngrid,temperature,pressure):
        orig=self.compute_opacity(temperature,pressure)
        return np.interp(wngrid,self.wavenumberGrid,orig)
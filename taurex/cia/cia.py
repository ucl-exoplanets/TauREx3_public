from taurex.log import Logger
import numpy as np

class CIA(Logger):
    """
    This is the base class for collisionally induced absorption opacities

    """
    
    def __init__(self,name,pair_name):
        super().__init__(name)

        self._pair_name = pair_name

    
    @property
    def pairName(self):
        return self._pair_name


    @property
    def pairOne(self):
        return self._pair_name.split('-')[0]

    @property
    def pairTwo(self):
        return self._pair_name.split('-')[-1]
    def compute_cia(self,temperature):
        raise NotImplementedError


    @property
    def wavenumberGrid(self):
        raise NotImplementedError

    @property
    def temperatureGrid(self):
        raise NotImplementedError


    def cia(self,temperature,wngrid=None):
        orig = self.compute_cia(temperature)
        if wngrid is None:
            return orig
        else:
            return np.interp(wngrid,self.wavenumberGrid,orig)

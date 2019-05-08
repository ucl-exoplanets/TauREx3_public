from taurex.log import Logger
from taurex.constants import G,RJUP,MJUP,RSOL
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.util.emission import black_body

class BlackbodyStar(Fittable,Logger):
    """Holds information on the star
    its default is a blackbody spectra

    Parameters
    -----------



    """
    

    def __init__(self,temperature=5000,radius=1.0):
        Logger.__init__(self,'Star')
        Fittable.__init__(self)
        self._temperature = temperature
        self._radius = radius*RSOL
        self._sed = None

    @property
    def radius(self):
        return self._radius
    
    @property
    def temperature(self):
        return self._temperature

    def initialize(self,wngrid):
        self.sed = black_body(wngrid,self.temperature)
    

    @property
    def spectralEmissionDensity(self):
        return self.sed

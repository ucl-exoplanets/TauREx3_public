from taurex.log import Logger
from taurex.constants import G,RJUP,MJUP,RSOL
from taurex.data.fittable import fitparam,Fittable
import numpy as np

class Star(Fittable,Logger):
    """Holds information on the star
    its default is a blackbody spectra

    Parameters
    -----------



    """
    

    def __init__(self,temperature=5000,radius=RSOL):
        Logger.__init__(self,'Star')
        Fittable.__init__(self)
        self._temperature = temperature
        self._radius = radius
    

    @property
    def radius(self):
        return self._radius
    
    @property
    def temperature(self):
        return self._temperature



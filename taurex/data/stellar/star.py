from taurex.log import Logger
from taurex.constants import G,RJUP,MJUP
from .fittable import fitparam,Fittable
import numpy as np

class Star(Fittable,Logger):
    """Holds information on the star
    its default is a blackbody spectra

    Parameters
    -----------



    """
    

    def __init__(self,name,temperature):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        self._temperature = temperature



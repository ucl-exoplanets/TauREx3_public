from taurex.log import Logger
from taurex.util import get_molecular_weight,molecule_texlabel
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.output.writeable import Writeable
import math

class Chemistry(Fittable,Logger,Writeable):
    """
    Defines Chemistry models

    """

    


    def __init__(self,name):
        Logger.__init__(self,name)
        Fittable.__init__(self)

    

    @property
    def activeGases(self):
        raise NotImplementedError
    

    @property
    def inactiveGases(self):
        raise NotImplementedError
    

    def initialize_chemistry(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        raise NotImplementedError


    @property
    def activeGasMixProfile(self):
        raise NotImplementedError

    @property
    def inactiveGasMixProfile(self):
        raise NotImplementedError


    @property
    def muProfile(self):
        raise NotImplementedError
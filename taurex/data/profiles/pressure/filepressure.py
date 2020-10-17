from .arraypressure import ArrayPressureProfile
from taurex.util.util import conversion_factor
import numpy as np


class FilePressureProfile(ArrayPressureProfile):

    def __init__(self, filename=None, usecols=0, skiprows=0, units='Pa',delimiter=None,reverse=False):
        
        to_Pa = conversion_factor(units, 'Pa')

        read_arr = np.loadtxt(filename, usecols=int(usecols), skiprows=int(skiprows),delimiter=delimiter,dtype=np.float64
                              )
        super().__init__(read_arr*to_Pa,reverse=reverse)


    @classmethod
    def input_keywords(self):
        return ['file', 'fromfile', 'loadfile',]


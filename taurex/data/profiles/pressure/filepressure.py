from .arraypressure import ArrayPressureProfile
from taurex.util.util import conversion_factor
import numpy as np


class FilePressureProfile(ArrayPressureProfile):

    def __init__(self, filename=None, usecols=0, skiprows=0, units='Pa'):
        
        to_Pa = conversion_factor(units, 'Pa')

        read_arr = np.loadtxt(filename, usecols=usecols, skiprows=skiprows,dtype=np.float64
                              )
        super().__init__(read_arr*to_Pa)


    @classmethod
    def input_keywords(self):
        return ['file', 'fromfile', 'loadfile',]


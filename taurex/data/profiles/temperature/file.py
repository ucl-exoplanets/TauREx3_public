from .temparray import TemperatureArray
import numpy as np
from taurex.util.util import conversion_factor


class TemperatureFile(TemperatureArray):
    """
    A temperature profile read from file

    Parameters
    ----------

    filename : str
        File name for temperature profile

    """

    def __init__(self, filename=None, skiprows=0, temp_col=0,
                 press_col=None, temp_units='K', press_units='Pa',delimiter=None, reverse=False):

        pressure_arr = None
        temperature_arr = None

        convertT = conversion_factor(temp_units, 'K')
        convertP = conversion_factor(press_units, 'Pa')

        if press_col is not None:
            arr = np.loadtxt(filename, skiprows=skiprows, 
                             usecols=(int(press_col), int(temp_col)),delimiter=delimiter,
                             )
            temperature_arr = arr[:, 1]*convertT
            pressure_arr = arr[:, 0]*convertP
        else:
            arr = np.loadtxt(filename, skiprows=skiprows,
                             usecols=int(temp_col),
                             )
            temperature_arr = arr[:]*convertT

        super().__init__(tp_array=temperature_arr, p_points=pressure_arr)


    @classmethod
    def input_keywords(cls):
        """
        Return all input keywords
        """
        return ['file','fromfile', ]
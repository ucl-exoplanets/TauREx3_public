from .temparray import TemperatureArray
import numpy as np


class TemperatureFile(TemperatureArray):
    """
    A temperature profile read from file

    Parameters
    ----------

    filename : str
        File name for temperature profile

    """

    def __init__(self, filename=None):
        super().__init__(tp_array=np.loadtxt(filename))

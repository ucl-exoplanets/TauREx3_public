from .tprofile import TemperatureProfile
import numpy as np


class TemperatureFile(TemperatureProfile):
    """A temperature profile read from file

    Parameters
    ----------
    
    filename : str
        File name for temperature profile

    """


    def __init__(self,filename=None):
        super().__init__(self.__class__.__name__)

        self._profile = np.loadtxt(filename)
    

    @property
    def profile(self):
        """Returns a temperature profile read from file

        Returns: :obj:np.array(float)
            temperature profile
        """



        return self._profile

    def write(self,output):
        temperature = super().write(output)
        return temperature
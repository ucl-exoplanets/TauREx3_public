from .tprofile import TemperatureProfile
import numpy as np


class TemperatureArray(TemperatureProfile):
    """

    Temperature profile loaded from array

    """

    def __init__(self, tp_array=[2000, 1000]):
        super().__init__(self.__class__.__name__)

        self._tp_profile = np.array(tp_array)

    @property
    def profile(self):
        """Returns an isothermal temperature profile

        Returns: :obj:np.array(float)
            temperature profile
        """

        if self._tp_profile.shape[0] == self.nlayers:
            return self._tp_profile
        else:
            interp_temp = np.linspace(0.0, 1.0, self._tp_profile.shape[0])
            interp_array = np.linspace(0.0, 1.0, self.nlayers)

            return np.interp(interp_array, interp_temp, self._tp_profile)

    def write(self, output):

        temperature = super().write(output)

        temperature.write_scalar('tp_array', self._tp_profile)

        return temperature

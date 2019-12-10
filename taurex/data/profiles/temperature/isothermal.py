from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam


class Isothermal(TemperatureProfile):
    """An isothermal temperature-pressure profile

    Parameters
    ----------

    T : float
        Isothermal Temperature to set

    """

    def __init__(self, T=1500):
        super().__init__('Isothermal')

        self._iso_temp = T

    @fitparam(param_name='T',
              param_latex='$T$',
              default_fit=False,
              default_bounds=[300.0, 2000.0])
    def isoTemperature(self):
        """Isothermal temperature in Kelvin"""
        return self._iso_temp

    @isoTemperature.setter
    def isoTemperature(self, value):
        self._iso_temp = value

    @property
    def profile(self):
        """Returns an isothermal temperature profile

        Returns: :obj:`array`
            temperature profile
        """

        T = np.zeros((self.nlayers))
        T[:] = self._iso_temp

        return T

    def write(self, output):
        temperature = super().write(output)
        temperature.write_scalar('T', self._iso_temp)
        return temperature

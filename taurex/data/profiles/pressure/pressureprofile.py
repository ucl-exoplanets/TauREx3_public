from taurex.log import Logger
from taurex.data.fittable import fitparam, Fittable
import numpy as np
from taurex.output.writeable import Writeable


class PressureProfile(Fittable, Logger, Writeable):
    """
    *Abstract Class*

    Base pressure class. Simple. Defines the layering
    of the atmosphere. Only requires
    implementation of:

    - :func:`compute_pressure_profile`
    - :func:`profile`

    Parameters
    ----------

    name: str
        Name used in logging

    nlayers: int
        Number of layers in atmosphere

    """

    def __init__(self, name, nlayers):
        Fittable.__init__(self)
        Logger.__init__(self, name)

        self._nlayers = int(nlayers)

    @property
    def nLayers(self):
        """
        Number of layers

        Returns
        -------
        int
        """

        return self._nlayers

    @property
    def nLevels(self):
        return self.nLayers + 1

    def compute_pressure_profile(self):
        """
        **Requires implementation**

        Compute pressure profile and
        generate pressure array in Pa

        Returns
        -------
        pressure_profile: :obj:`array`
            Pressure profile array in Pa

        """
        raise NotImplementedError

    @property
    def profile(self):
        """
        Returns pressure at each atmospheric layer (Pascal)

        Returns
        -------
        pressure_profile : :obj:`array`
        """
        raise NotImplementedError

    def write(self, output):
        pressure = output.create_group('Pressure')
        pressure.write_string('pressure_type', self.__class__.__name__)
        pressure.write_scalar('nlayers', self._nlayers)
        pressure.write_array('profile', self.profile)
        return pressure


class SimplePressureProfile(PressureProfile):
    """
    A basic pressure profile.

    Parameters
    ----------
    nlayers : int
        Number of layers in atmosphere

    atm_min_pressure : float
        minimum pressure in Pascal (top of atmosphere)

    atm_max_pressure : float
        maximum pressure in Pascal (surface of planet)

    """

    def __init__(self, nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6):

        super().__init__('pressure_profile', nlayers)
        self.pressure_profile = None
        self._atm_min_pressure = atm_min_pressure
        self._atm_max_pressure = atm_max_pressure

    def compute_pressure_profile(self):
        """
        Sets up the pressure profile for the atmosphere model

        """

        # set pressure profile of layer boundaries
        press_exp = np.linspace(np.log(self._atm_min_pressure),
                                np.log(self._atm_max_pressure),
                                self.nLevels)
        self.pressure_profile_levels = np.exp(press_exp)[::-1]

        # get mid point pressure between levels (i.e. get layer pressure)
        # computing geometric
        # average between pressure at n and n+1 level
        self.pressure_profile = \
            np.power(10, np.log10(self.pressure_profile_levels)[:-1] +
                     np.diff(np.log10(self.pressure_profile_levels))/2.)

    @fitparam(param_name='atm_min_pressure',
              param_latex='$P_\mathrm{min}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[0.1, 1.0])
    def minAtmospherePressure(self):
        """Minimum pressure of atmosphere (top layer) in Pascal"""
        return self._atm_min_pressure

    @minAtmospherePressure.setter
    def minAtmospherePressure(self, value):
        self._atm_min_pressure = value

    @fitparam(param_name='atm_max_pressure',
              param_latex='$P_\\mathrm{max}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[0.1, 1.0])
    def maxAtmospherePressure(self):
        """Maximum pressure of atmosphere (surface) in Pascal"""

        return self._atm_max_pressure

    @maxAtmospherePressure.setter
    def maxAtmospherePressure(self, value):
        self._atm_max_pressure = value

    @property
    def profile(self):
        return self.pressure_profile

    def write(self, output):
        pressure = super().write(output)

        pressure.write_scalar('atm_max_pressure', self._atm_max_pressure)
        pressure.write_scalar('atm_min_pressure', self._atm_min_pressure)

        return pressure

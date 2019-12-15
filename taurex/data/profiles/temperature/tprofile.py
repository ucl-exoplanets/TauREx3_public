from taurex.log import Logger
from taurex.data.fittable import Fittable
from taurex.output.writeable import Writeable


class TemperatureProfile(Fittable, Logger, Writeable):
    """

    *Abstract Class*

    Defines temperature profile for an atmosphere

    Must define:

    - :func:`profile`

    Parameters
    ----------
    name : str
        Name used in logging

    """

    def __init__(self, name):
        Logger.__init__(self, name)
        Fittable.__init__(self)

    def initialize_profile(self, planet=None, nlayers=100,
                           pressure_profile=None):
        """
        Initializes the profile

        Parameters
        ----------
        planet: :class:`~taurex.data.planet.Planet`

        nlayers: int
            Number of layers in atmosphere

        pressure_profile: :obj:`array`
            Pressure at each layer of the atmosphere

        """
        self.nlayers = nlayers
        self.nlevels = nlayers+1
        self.pressure_profile = pressure_profile
        self.planet = planet

    @property
    def profile(self):
        """
        Must return a temperature profile at each layer of the atmosphere

        Returns
        -------
        temperature: :obj:`array`
            Temperature in Kelvin
        """
        raise NotImplementedError

    def write(self, output):
        temperature = output.create_group('Temperature')
        temperature.write_string('temperature_type', self.__class__.__name__)

        return temperature

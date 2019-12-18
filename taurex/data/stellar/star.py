from taurex.log import Logger
from taurex.constants import G, RJUP, MJUP, RSOL, MSOL
from taurex.data.fittable import fitparam, Fittable
import numpy as np
from taurex.util.emission import black_body
from taurex.output.writeable import Writeable


class Star(Fittable, Logger, Writeable):
    """
    A base class that holds information on the star in the model.
    Its implementation is a star that has a blackbody spectrum.


    Parameters
    ----------

    temperature: float, optional
        Stellar temperature in Kelvin

    radius: float, optional
        Stellar radius in Solar radius

    metallicity: float, optional
        Metallicity in solar values

    mass: float, optional
        Stellar mass in solar mass

    distance: float, optional
        Distance from Earth in pc

    magnitudeK: float, optional
        Maginitude in K band


    """

    def __init__(self, temperature=5000, radius=1.0, distance=1,
                 magnitudeK=10.0, mass=1.0, metallicity=1.0):

        Logger.__init__(self, self.__class__.__name__)
        Fittable.__init__(self)
        self._temperature = temperature
        self._radius = radius*RSOL
        self._mass = mass*MSOL
        self.debug('Star mass %s', self._mass)
        self.sed = None
        self.distance = distance
        self.magnitudeK = magnitudeK
        self._metallicity = metallicity

    @property
    def radius(self):
        """
        Radius in metres

        Returns
        -------
        R: float

        """
        return self._radius

    @property
    def temperature(self):
        """
        Blackbody temperature in Kelvin

        Returns
        -------
        T: float

        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def mass(self):
        return self._mass

    def initialize(self, wngrid):
        """
        Initializes the blackbody spectrum on the given wavenumber grid

        Parameters
        ----------
        wngrid: :obj:`array`
            Wavenumber grid cm-1 to compute black body spectrum

        """
        self.sed = black_body(wngrid, self.temperature)

    @property
    def spectralEmissionDensity(self):
        """
        Spectral emmision density

        Returns
        -------
        sed: :obj:`array`
        """
        return self.sed

    def write(self, output):
        star = output.create_group('Star')
        star.write_string('star_type', self.__class__.__name__)
        star.write_scalar('temperature', self.temperature)
        star.write_scalar('radius', self._radius/RSOL)
        star.write_scalar('distance', self.distance)
        star.write_scalar('mass', self._mass/MSOL)
        star.write_scalar('magnitudeK', self.magnitudeK)
        star.write_scalar('metallicity', self._metallicity)
        star.write_scalar('radius_m', self.radius)
        star.write_array('SED', self.spectralEmissionDensity)
        star.write_scalar('mass_kg', self._mass)
        return star


class BlackbodyStar(Star):
    """Alias for the base star type"""
    pass

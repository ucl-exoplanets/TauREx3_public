from taurex.log import Logger
from taurex.constants import G, RJUP, MJUP, AU
from .fittable import fitparam, Fittable
from taurex.output.writeable import Writeable


class Planet(Fittable, Logger, Writeable):
    """Holds information on a planet and its properties and
    derived properties

    Parameters
    -----------

    planet_mass: float, optional
        mass in terms of Jupiter mass of the planet
    planet_radius: float, optional
        radius in terms of Jupiter radii of the planet
    planet_distance: float, optional
        Semi-major axis in AU
    impact_param: float, optional
        Impact parameter
    orbital_period: float, optional
        Orbital period in days
    albedo: float, optional
        Planetary albedo
    transit_time: float, optional
        Transit time in seconds

    """

    def __init__(self, planet_mass=1.0, planet_radius=1.0,
                 planet_distance=1,
                 impact_param=0.5, orbital_period=2.0, albedo=0.3,
                 transit_time=3000.0):
        Logger.__init__(self, 'Planet')
        Fittable.__init__(self)
        self._mass = planet_mass*MJUP
        self._radius = planet_radius*RJUP
        self._distance = planet_distance*AU
        self._impact = impact_param
        self._orbit_period = orbital_period
        self._albedo = albedo
        self._transit_time = transit_time

    @fitparam(param_name='planet_mass', param_latex='$M_p$',
              default_fit=False, default_bounds=[0.5, 1.5])
    def mass(self):
        """
        Planet mass in Jupiter mass
        """
        return self._mass/MJUP

    @mass.setter
    def mass(self, value):
        self._mass = value*MJUP

    @fitparam(param_name='planet_radius', param_latex='$R_p$',
              default_fit=True, default_bounds=[0.9, 1.1])
    def radius(self):
        """
        Planet radius in Jupiter radii
        """
        return self._radius/RJUP

    @radius.setter
    def radius(self, value):
        self._radius = value*RJUP

    @property
    def fullRadius(self):
        """
        Planet radius in metres
        """
        return self._radius

    @property
    def fullMass(self):
        """
        Planet mass in kg
        """
        return self._mass

    @property
    def impactParameter(self):
        return self._impact

    @property
    def orbitalPeriod(self):
        return self._orbit_period

    @property
    def albedo(self):
        return self._albedo

    @property
    def transitTime(self):
        return self._transit_time

    @fitparam(param_name='planet_distance', param_latex='$D_{planet}$',
              default_fit=False, default_bounds=[1, 2])
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value

    @property
    def gravity(self):
        """
        Surface gravity in ms-2
        """
        return (G * self.fullMass) / (self.fullRadius**2)

    def gravity_at_height(self, height):
        """
        Gravity at height (m) from planet in ms-2

        Parameters
        ----------
        height: float
            Height in metres from planet surface

        Returns
        -------
        g: float
            Gravity in ms-2

        """

        return (G * self.fullMass) / ((self.fullRadius+height)**2)

    def write(self, output):
        planet = output.create_group('Planet')

        planet.write_string('planet_type', self.__class__.__name__)
        planet.write_scalar('planet_mass', self._mass/MJUP)
        planet.write_scalar('planet_radius', self._radius/RJUP)
        planet.write_scalar('planet_distance', self._distance/AU)
        planet.write_scalar('impact_param', self._impact)
        planet.write_scalar('orbital_period', self.orbitalPeriod)
        planet.write_scalar('albedo', self.albedo)
        planet.write_scalar('transit_time', self.transitTime)

        planet.write_scalar('mass_kg', self.mass)
        planet.write_scalar('radius_m', self.radius)
        planet.write_scalar('surface_gravity', self.gravity)
        return planet


class Earth(Planet):
    """An implementation for earth"""

    def __init__(self):
        Planet.__init__(self, 5.972e24, 6371000)


class Mars(Planet):

    def __init__(self):
        import astropy.units as u

        radius = (0.532*u.R_earth).to(u.jupiterRad)
        mass = (0.107*u.M_earth).to(u.jupiterMass)
        distance = 1.524
        Planet.__init__(mass=mass.value, radius=radius.value,
                        distance=distance)

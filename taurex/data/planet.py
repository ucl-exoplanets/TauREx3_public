from taurex.log import Logger
from taurex.constants import G, RJUP, MJUP, AU
from .fittable import fitparam, Fittable, derivedparam
from taurex.output.writeable import Writeable
import math
from .citation import Citable
from taurex.util.util import conversion_factor

class BasePlanet(Fittable, Logger, Writeable, Citable):
    """Holds information on a planet and its properties and
    derived properties

    Parameters
    -----------

    planet_mass: float, optional
        mass in terms of Jupiter mass of the planet
    planet_radius: float, optional
        radius in terms of Jupiter radii of the planet
    planet_sma: float, optional
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
                 planet_sma=None, planet_distance=1.0,
                 impact_param=0.5, orbital_period=2.0, albedo=0.3,
                 transit_time=3000.0):
        Logger.__init__(self, 'Planet')
        Fittable.__init__(self)
        self.set_planet_mass(planet_mass, 'Mjup')
        self.set_planet_radius(planet_radius, 'Rjup')
        self.set_planet_semimajoraxis(planet_sma or planet_distance)
        self._impact = impact_param
        self._orbit_period = orbital_period
        self._albedo = albedo
        self._transit_time = transit_time

    def set_planet_radius(self, value, unit='Rjup'):
        factor = conversion_factor(unit, 'm')
        self._radius = value*factor

    def set_planet_mass(self, value, unit='Mjup'):
        factor = conversion_factor(unit, 'kg')
        self._mass = value*factor

    def set_planet_semimajoraxis(self, value, unit='AU'):
        factor = conversion_factor(unit, 'm')
        self._distance = value*factor

    def get_planet_radius(self, unit='Rjup'):
        factor = conversion_factor('m', unit)
        return self._radius*factor
    
    def get_planet_mass(self, unit='Mjup'):
        factor = conversion_factor('kg', unit)
        return self._mass*factor

    def get_planet_semimajoraxis(self, unit='AU'):
        factor = conversion_factor('m', unit)
        return self._distance*factor


    @fitparam(param_name='planet_mass', param_latex='$M_p$',
              default_fit=False, default_bounds=[0.5, 1.5])
    def mass(self):
        """
        Planet mass in Jupiter mass
        """
        return self.get_planet_mass(unit='Mjup')

    @mass.setter
    def mass(self, value):
        self.set_planet_mass(value, unit='Mjup')

    @fitparam(param_name='planet_radius', param_latex='$R_p$',
              default_fit=True, default_bounds=[0.9, 1.1])
    def radius(self):
        """
        Planet radius in Jupiter radii
        """
        return self.get_planet_radius(unit='Rjup')

    @radius.setter
    def radius(self, value):
        self.set_planet_radius(value, unit='Rjup')

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
        """
        Planet semi major axis from parent star (AU)
        """
        return self.get_planet_semimajoraxis(unit='AU')

    @distance.setter
    def distance(self, value):
        self.set_planet_semimajoraxis(value, unit='AU')

    @fitparam(param_name='planet_sma', param_latex='$D_{planet}$',
              default_fit=False, default_bounds=[1, 2])
    def semiMajorAxis(self):
        """
        Planet semi major axis from parent star (AU) (ALIAS)
        """
        return self.get_planet_semimajoraxis(unit='AU')

    @semiMajorAxis.setter
    def semiMajorAxis(self, value):
        self.set_planet_semimajoraxis(value, unit='AU')


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

    @derivedparam(param_name='logg', param_latex='log(g)', compute=False)
    def logg(self):
        """
        Surface gravity (m2/s) in log10
        """ 
        return math.log10(self.gravity)

    def calculate_scale_properties(self, T, Pl, mu, length_units='m'):
        """
        Computes altitude, gravity and scale height of the atmosphere.


        Parameters
        ----------
        T: array_like
            Temperature of each layer in K

        Pl: array_like
            Pressure at each layer boundary in Pa

        mu: array_like
            mean moleculer weight for each layer in kg

        Returns
        -------
        z: array
            Altitude at each layer boundary
        H: array
            scale height converted to correct length units
        g: array
            gravity converted to correct length units
        deltaz:
            dz in length units

        """
        import numpy as np
        from taurex.constants import KBOLTZ
        from taurex.util.util import conversion_factor
        # build the altitude profile from the bottom up
        nlayers = T.shape[0]
        H = np.zeros(nlayers)
        g = np.zeros(nlayers)
        z = np.zeros(nlayers+1)
        deltaz = np.zeros(nlayers+1)

        # surface gravity (0th layer)
        g[0] = self.gravity
        # scaleheight at the surface (0th layer)
        H[0] = (KBOLTZ*T[0])/(mu[0]*g[0])
        #####
        ####
        ####

        factor = conversion_factor('m', length_units)

        for i in range(1, nlayers+1):
            deltaz[i] = (-1.)*H[i-1]*np.log(Pl[i]/Pl[i-1])
            z[i] = z[i-1] + deltaz[i]
            if i < nlayers:
                with np.errstate(over='ignore'):
                    # gravity at the i-th layer
                    g[i] = self.gravity_at_height(z[i])
                    self.debug('G[%s] = %s', i, g[i])

                with np.errstate(divide='ignore'):
                    H[i] = (KBOLTZ*T[i])/(mu[i]*g[i])

        return z*factor, H*factor, g*factor, deltaz[1:]*factor

    def compute_path_length(self, altitudes, viewer, tangent,
                            vector_coord_sys='cartesian'):
        from taurex.util.geometry import compute_path_length_3d

        result = compute_path_length_3d(self.fullRadius,
                                        altitudes, viewer, tangent,
                                        coordinates=vector_coord_sys)

        return result

    @classmethod
    def input_keywords(self):
        raise NotImplementedError


class Planet(BasePlanet):
    @classmethod
    def input_keywords(self):
        return ['simple', 'sphere']



class Earth(Planet):
    """An implementation for earth"""

    def __init__(self):
        Planet.__init__(self, 5.972e24, 6371000)

    @classmethod
    def input_keywords(self):
        return ['earth', ]

class Mars(Planet):

    def __init__(self):
        import astropy.units as u

        radius = (0.532*u.R_earth).to(u.jupiterRad)
        mass = (0.107*u.M_earth).to(u.jupiterMass)
        distance = 1.524
        Planet.__init__(mass=mass.value, radius=radius.value,
                        distance=distance)

    @classmethod
    def input_keywords(self):
        return ['mars', ]
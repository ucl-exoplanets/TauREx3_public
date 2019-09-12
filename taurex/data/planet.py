from taurex.log import Logger
from taurex.constants import G,RJUP,MJUP
from .fittable import fitparam,Fittable
import numpy as np
from taurex.output.writeable import Writeable
class Planet(Fittable,Logger,Writeable):
    """Holds information on a planet and its properties and 
    derived properties

    Parameters
    -----------

    mass: float
        mass in terms of Jupiter mass of the planet
    radius : float
        radius in terms of Jupiter radii of the planet

    """
    
    def __init__(self,mass=1.0,radius=1.0,
                      distance=1,
                      impact_param=0.5,orbital_period=2.0, albedo=0.3,
                      transit_time=3000.0):
        Logger.__init__(self,'Planet')
        Fittable.__init__(self)
        self._mass = mass*MJUP
        self._radius = radius*RJUP
        self._distance = distance
        self._impact = impact_param
        self._orbit_period = orbital_period
        self._albedo = albedo
        self._transit_time = transit_time

    
    @fitparam(param_name='planet_mass',param_latex='$M_p$',default_fit=False,default_bounds=[0.5,1.5])
    def mass(self):
        """
        Planet mass in Jupiter mass
        """
        return self._mass/MJUP
    
    @mass.setter
    def mass(self,value):
        self._mass = value*MJUP


    @fitparam(param_name='planet_radius',param_latex='$R_p$',default_fit=True,default_bounds=[0.9,1.1])
    def radius(self):
        """
        Planet radius in Jupiter radii
        """
        return self._radius/RJUP
    
    @radius.setter
    def radius(self,value):
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


    # @fitparam(param_name='planet_ld_coeff',param_latex=None,default_fit=False)
    # def limbDarkeningCoeff(self):
    #     return self._ld_coeff
    
    # @limbDarkeningCoeff.setter
    # def limbDarkeningCoeff(self,value):
    #     self._ld_coeff = value

    @fitparam(param_name='planet_distance',param_latex='$D_{planet}$',default_fit=False,default_bounds=[1,2])
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self,value):
        self._distance = value




    @property
    def gravity(self):
        """
        Surface gravity in ms-2
        """
        return (G * self.fullMass) / (self.fullRadius**2) 

    
    def gravity_at_height(self,height):
        """
        Gravity at height (m) from planet in ms-2

        Parameters
        ----------
        height : float
            Height in metres from planet surface
        
        Returns
        -------
        g : float

        """

        return (G * self.fullMass) / ((self.fullRadius+height)**2) 

    def write(self,output):
        planet = output.create_group('Planet')
        planet.write_string('planet_type',self.__class__.__name__)
        planet.write_scalar('mass',self._mass)
        planet.write_scalar('radius',self._radius)
        planet.write_scalar('mass_MJUP',self.mass)
        planet.write_scalar('radius_RJUP',self.radius)
        planet.write_scalar('distance',self._distance)
        planet.write_scalar('surface_gravity',self.gravity)
        return planet

class Earth(Planet):
    """An implementation for earth"""
    def __init__(self):
        Planet.__init__(self,5.972e24,6371000)


class Mars(Planet):

    def __init__(self):
        import astropy.units as u

        radius = (0.532*u.R_earth).to(u.jupiterRad)
        mass = (0.107*u.M_earth).to(u.jupiterMass)
        distance=1.524
        Planet.__init__(mass=mass.value,radius=radis.value,distance=distance)
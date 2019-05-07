from taurex.log import Logger
from taurex.constants import G,RJUP,MJUP
from .fittable import fitparam,Fittable
import numpy as np

class Planet(Fittable,Logger):
    """Holds information on a planet and its properties and 
    derived properties

    Parameters
    -----------

    mass: float
        mass in kg of the planet
    radius
        radius of the planet in meters

    """
    
    def __init__(self,mass=1.0,radius=1.0,ld_coeff=1.0,distance=1):
        Logger.__init__(self,'Planet')
        Fittable.__init__(self)
        self._mass = mass*MJUP
        self._radius = radius*RJUP
        self._ld_coeff = ld_coeff
        self._distance = distance

    
    @fitparam(param_name='planet_mass',param_latex='$M_p$',default_fit=False,default_bounds=[0.5,1.5])
    def mass(self):
        return self._mass/MJUP
    
    @mass.setter
    def mass(self,value):
        self._mass = value*MJUP


    @fitparam(param_name='planet_radius',param_latex='$R_p$',default_fit=True,default_bounds=[0.9,1.1])
    def radius(self):
        return self._radius/RJUP
    
    @radius.setter
    def radius(self,value):
        self._radius = value*RJUP
    
    @property
    def fullRadius(self):
        return self._radius

    @property
    def fullMass(self):
        return self._mass

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
        return (G * self.fullMass) / (self.fullRadius**2) 

    
    def gravity_at_height(self,height):
        return (G * self.fullMass) / ((self.fullRadius+height)**2) 



class Earth(Planet):
    """An implementation for earth"""
    def __init__(self):
        Planet.__init__(self,5.972e24,6371000)
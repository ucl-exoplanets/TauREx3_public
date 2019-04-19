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
    
    def __init__(self,mass=RJUP,radius=MJUP,ld_coeff=1.0,distance=1):
        Logger.__init__(self,'Planet')
        Fittable.__init__(self)
        self._mass = mass
        self._radius = radius
        self._ld_coeff = ld_coeff
        self._distance = distance

    
    @fitparam(param_name='planet_mass',param_latex='$M_p$',default_fit=False)
    def mass(self):
        return self._mass
    
    @mass.setter
    def mass(self,value):
        self._mass = value


    @fitparam(param_name='planet_radius',param_latex='$R_p$',default_fit=True,default_bounds=[0.9*RJUP,1.1*RJUP])
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self,value):
        self._radius = value

    @fitparam(param_name='planet_ld_coeff',param_latex=None,default_fit=False)
    def limbDarkeningCoeff(self):
        return self._ld_coeff
    
    @limbDarkeningCoeff.setter
    def limbDarkeningCoeff(self,value):
        self._ld_coeff = value

    @fitparam(param_name='planet_distance',param_latex='$D_{planet}$',default_fit=False,default_bounds=[1,2])
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self,value):
        self._distance = value


    @property
    def gravity(self):
        return (G * self.mass) / (self.radius**2) 



class Earth(Planet):
    """An implementation for earth"""
    def __init__(self):
        Planet.__init__(self,5.972e24,6371000)
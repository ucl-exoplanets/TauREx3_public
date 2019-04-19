from taurex.log import Logger
from taurex.constants import G
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
    
    def __init__(self,mass,radius):
        Logger.__init__(self,'Planet')
        Fittable.__init__(self)
        self._mass = mass
        self._radius = radius

    
    @fitparam(param_name='planet_mass',param_latex=None)
    def mass(self):
        return self._mass
    
    @mass.setter
    def mass(self,value):
        self._mass = value


    @fitparam(param_name='planet_radius',param_latex=None)
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self,value):
        self._radius = value
        

    @property
    def gravity(self):
        return (G * self.mass) / (self.radius**2) 



class Earth(Planet):
    """An implementation for earth"""
    def __init__(self):
        Planet.__init__(self,5.972e24,6371000)
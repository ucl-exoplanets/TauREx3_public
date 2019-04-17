from taurex.log import Logger
from taurex.constants import G
import numpy as np

class Planet(Logger):
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
        super().__init__('Planet')
        self._mass = mass
        self._radius = radius

    
    @property
    def mass(self):
        return self._mass
    
    @property
    def radius(self):
        return self._radius
        

    @property
    def gravity(self):
        return (G * self.mass) / (self.radius**2) 



class Earth(Planet):
    """An implementation for earth"""
    def __init__(self):
        super().__init__(5.972e24,6371000)
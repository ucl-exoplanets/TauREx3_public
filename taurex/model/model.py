from taurex.log import Logger
import numpy as np
import math

class ForwardModel(Logger):
    """A base class for producing forward models"""

    def __init__(self,name):
        super().__init__(name)

    
    


    def update_model(self,wn_grid):
        """Recomputes the model for a specifc wngrid"""
        raise NotImplementedError

    
    def model(self):
        """Returns the currently computed model"""
        raise NotImplementedError

    


class SimpleForwardModel(ForwardModel):

    def __init__(self,name,
                            planet=None,
                            star=None,
                            pressure_profile=None,
                            temperature_profile=None,
                            gas_profile=None
                            ):
        super().__init__(name)

        self._planet = planet
        self._star = star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._gas_profile = gas_profile


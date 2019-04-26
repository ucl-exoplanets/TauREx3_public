from taurex.log import Logger
import numpy as np
import math

class ForwardModel(Logger):
    """A base class for producing forward models"""

    def __init__(self,name):
        super().__init__(name)

    
    


    def model(self,wn_grid):
        """Computes the forward model for a wngrid"""
        raise NotImplementedError

    


class SimpleForwardModel(ForwardModel):
    """ A 'simple' base model in the sense that its just
    a fairly standard single profiles model. Most like you'll
    inherit from this to do your own fuckery
    
    Parameters
    ----------
    name: string
        Name to use in logging
    
    planet: :obj:`Planet` or :obj:`None`
        Planet object created or None to use a default

    
    """
    def __init__(self,name,
                            planet=None,
                            star=None,
                            pressure_profile=None,
                            temperature_profile=None,
                            gas_profile=None,
                            opacities=None,
                            atm_min_pressure=
                            ):
        super().__init__(name)

        self._planet = planet
        self._star = star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._gas_profile = gas_profile


    
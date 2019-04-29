from taurex.log import Logger
import numpy as np
import math
from .simplemodel import SimpleForwardModel


class TransmissionModel(SimpleForwardModel):
    """
    Parameters
    ----------
    name: string
        Name to use in logging
    
    planet: :obj:`Planet` or :obj:`None`
        Planet object created or None to use the default planet (Jupiter)

    
    """
    def __init__(self,
                            planet=None,
                            star=None,
                            pressure_profile=None,
                            temperature_profile=None,
                            gas_profile=None,
                            opacities=None,
                            cia=None,
                            opacity_path=None,
                            cia_path=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,

                            ):
        super().__init__('transmission_model',planet,
                            star,
                            pressure_profile,
                            temperature_profile,
                            gas_profile,
                            opacities,
                            cia,
                            opacity_path,
                            cia_path,
                            nlayers,
                            atm_min_pressure,
                            atm_max_pressure)
    
    def path_integral(self,wngrid):


        absorption = np.zeros_like(wngrid)

        
        
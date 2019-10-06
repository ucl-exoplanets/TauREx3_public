from taurex.log import Logger
import numpy as np
import math
from .simplemodel import SimpleForwardModel
from .emission import EmissionModel
from taurex.constants import PI
class DirectImageModel(EmissionModel):
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
                            chemistry=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,
                            ngauss=4,
                            ):
        super().__init__(planet,
                            star,
                            pressure_profile,
                            temperature_profile,
                            chemistry,
                            nlayers,
                            atm_min_pressure,
                            atm_max_pressure,
                            ngauss=ngauss)
    

    def compute_final_flux(self,f_total):
        star_distance_meters = self._star.distance*3.08567758e16

        SDR = pow((star_distance_meters/3.08567758e16),2)
        SDR = 1.0
        planet_radius = self._planet.fullRadius
        return((f_total * (planet_radius**2) * 2.0 * PI) / (4 * PI * (star_distance_meters**2))) * SDR
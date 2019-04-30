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
    
    def path_integral(self):

        dz=np.gradient(self.altitude_profile)
        wngrid_size=self.sigma_xsec.shape[-1]
        tau = np.zeros(shape=(self.nLayers,len(wngrid_size),))

        active_gas = self._gas_profile.activeGasMixProfile.transpose()
        density_profile = self.densityProfile

        total_layers = self.nLayers
        
        for layer in range(total_layers):
            
            tau[layer] = np.sum(
                          np.sum(self.sigma_xsec[layer:total_layers,:]* \
                                 active_gas[layer:total_layers,:,None]* \
                                 density_profile[layer:total_layers,None],axis=0),axis=2)
        
        tau = np.exp(-tau)
        integral = (self._planet.radius+self.altitude_profile[:,None])*(1.0-tau)*dz[:,None]
        integral*=2.0
        integral = np.sum(integral,axis=0)

        absorption = ((self._planet.radius**2.0) + integral)/(self._star.radius**2)
        return absorption,tau
        







        
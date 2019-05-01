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
    

    def compute_path_length(self,dz):


        dl = []


        planet_radius =self._planet.radius
        total_layers = self.nLayers

        z = self.altitudeProfile
        self.debug('Computing path_length: \n z={} \n dz={}'.format(z,dz))
        for layer in range(0,total_layers):

            p = (self._planet.radius+dz[0]/2 + z[layer])**2
            k = np.zeros(shape=(self.nLayers-layer))
            k[0] = 2.0 * np.sqrt((planet_radius + dz[0]/2. + z[layer] + dz[layer]/2.)**2 - p)

            k[1:]= np.sqrt((planet_radius + dz[0]/2 + z[layer+1:] + dz[layer+1:]/2)**2 - p) 
            k[1:] -= np.sqrt((planet_radius + dz[0]/2 + z[layer:self.nLayers-1] + dz[layer:self.nLayers-1]/2)**2 -p)

            dl.append(k)
        return dl
    def path_integral(self):

        dz=np.gradient(self.altitudeProfile)
        
        wngrid_size=self.sigma_xsec.shape[-1]
        
        path_length = self.compute_path_length(dz)

        tau = np.zeros(shape=(self.nLayers,wngrid_size,))

        active_gas = self._gas_profile.activeGasMixProfile.transpose()
        density_profile = self.densityProfile

        total_layers = self.nLayers
        path_length = self.compute_path_length(dz)

        for layer in range(total_layers):

            self.debug('Computing layer {}'.format(layer))
            dl = path_length[layer]
            comp = self.sigma_xsec[layer:total_layers,:]* \
                                            active_gas[layer:total_layers,:,None]* \
                                            density_profile[layer:total_layers,None,None]*dl[:,None,None]
            self.debug('Compoutation result {}'.format(comp))
            self.debug('Shape = {}'.format(comp.shape))

            comp = np.sum(comp,axis=0)
            self.debug('Sum Shape = {}'.format(comp.shape))

            comp = np.sum(comp,axis=0)
            self.debug('Post Sum Shape = {}'.format(comp.shape))

            tau[layer,...] = comp[...]
        
        tau = np.exp(-tau)
        integral = (self._planet.radius+self.altitudeProfile[:,None])*(1.0-tau)*dz[:,None]
        integral*=2.0
        integral = np.sum(integral,axis=0)

        absorption = ((self._planet.radius**2.0) + integral)/(self._star.radius**2)
        return absorption,tau
        







        
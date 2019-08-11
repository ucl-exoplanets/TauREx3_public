from taurex.log import Logger
import numpy as np
import math
from .simplemodel import SimpleForwardModel
from taurex.contributions import AbsorptionContribution
from taurex.constants import *
from taurex.util.emission import black_body
import numexpr as ne
class EmissionModel(SimpleForwardModel):
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
                            atm_max_pressure=1e6
                            ):
        super().__init__('emission_model',planet,
                            star,
                            pressure_profile,
                            temperature_profile,
                            chemistry,
                            nlayers,
                            atm_min_pressure,
                            atm_max_pressure)



    def compute_final_flux(self,f_total):
        star_sed = self._star.spectralEmissionDensity

        self.debug('Star SED: %s',star_sed)
        #quit()
        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius
        return (f_total/star_sed) * (planet_radius/star_radius)**2




    def path_integral(self,wngrid,return_contrib):
        import numexpr as ne

        dz=np.gradient(self.altitudeProfile)
        

        density= self.densityProfile

        wngrid_size=wngrid.shape[0]
        
        total_layers = self.nLayers

        mu1 = 0.1834346
        mu2 = 0.5255324
        mu3 = 0.7966665
        mu4 = 0.9602899

        w1 = 0.3626838
        w2 = 0.3137066
        w3 = 0.2223810
        w4 = 0.1012885

        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers,wngrid_size))
        surface_tau = np.zeros(shape=(1,wngrid_size))

        layer_tau = np.zeros(shape=(1,wngrid_size))

        dtau = np.zeros(shape=(1,wngrid_size))

        #Do surface first
        #for layer in range(total_layers):
        for contrib in self.contribution_list:
            contrib.contribute(self,0,total_layers,0,0,density,surface_tau,path_length=dz)
        self.debug('density = %s',density[0])
        self.debug('surface_tau = %s',surface_tau)

        BB = black_body(wngrid,temperature[0])/PI

        I1 = ne.evaluate('BB * ( exp(-surface_tau/mu1))')
        I2 = ne.evaluate('BB * ( exp(-surface_tau/mu2))')
        I3 = ne.evaluate('BB * ( exp(-surface_tau/mu3))')
        I4 = ne.evaluate('BB * ( exp(-surface_tau/mu4))')
        self.debug('I1_pre %s',I1)
        #Loop upwards
        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in self.contribution_list:
                contrib.contribute(self,layer+1,total_layers,0,0,density,layer_tau,path_length=dz)  
                contrib.contribute(self,layer,layer+1,0,0,density,dtau,path_length=dz)           

            _tau = ne.evaluate('exp(-layer_tau) - exp(-dtau)')

            tau[layer] += _tau[0]
            #for contrib in self.contribution_list:

            self.debug('Layer_tau[%s]=%s',layer,layer_tau)
            
            dtau += layer_tau
            self.debug('dtau[%s]=%s',layer,dtau)
            BB = black_body(wngrid,temperature[layer])/PI
            self.debug('BB[%s]=%s,%s',layer,temperature[layer],BB)
            I1 += ne.evaluate('BB * ( exp(-layer_tau/mu1) - exp(-dtau/mu1))')
            I2 += ne.evaluate('BB * ( exp(-layer_tau/mu2) - exp(-dtau/mu2))')
            I3 += ne.evaluate('BB * ( exp(-layer_tau/mu3) - exp(-dtau/mu3))')
            I4 += ne.evaluate('BB * ( exp(-layer_tau/mu4) - exp(-dtau/mu4))')            

        
        self.debug('I1: %s',I1)
        flux_total = ne.evaluate('2.0*PI*(I1*mu1*w1 + I2*mu2*w2 + I3*mu3*w3 + I4*mu4*w4)')
        self.debug('flux_total %s',flux_total)
        
        return self.compute_final_flux(flux_total).flatten(),tau
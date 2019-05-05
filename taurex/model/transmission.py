from taurex.log import Logger
import numpy as np
import math
from .simplemodel import SimpleForwardModel
from taurex.contributions import AbsorptionContribution

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
                            opacity_path=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,
                            abs_contrib= None
                            ):
        super().__init__('transmission_model',planet,
                            star,
                            pressure_profile,
                            temperature_profile,
                            gas_profile,
                            opacities,
                            opacity_path,
                            nlayers,
                            atm_min_pressure,
                            atm_max_pressure)

        if abs_contrib is None:
            abs_contrib = AbsorptionContribution()
        self.add_contribution(abs_contrib)

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



    def path_integral(self,wngrid,return_contrib):
        import numexpr as ne

        dz=np.gradient(self.altitudeProfile)
        
        wngrid_size=wngrid.shape[0]
        
        path_length = self.compute_path_length(dz)


        density_profile = self.densityProfile

        total_layers = self.nLayers

        path_length = self.compute_path_length(dz)
        self.path_length = path_length
        


        for layer in range(total_layers):

            self.debug('Computing layer {}'.format(layer))
            dl = path_length[layer]


            for contrib in self.contribution_list:
                self.debug('Adding contribution from {}'.format(contrib.name))
                contrib.contribute(self,layer,density_profile,dl,return_contrib)

        all_contrib = [c.totalContribution for c in self.contribution_list]

        self.debug('Contributionlist {}'.format(all_contrib))
        tau = np.sum(all_contrib,axis=0)
        self.debug('tau {} {}'.format(tau,tau.shape))

        absorption,tau = self.compute_absorption(tau,dz)

        contrib_absorption = []
        if return_contrib:
            contrib_absorption = []
            for contrib in self.contribution_list:
                c_tau = contrib.totalContribution
                contrib_absorption.append((contrib.name,self.compute_absorption(c_tau,dz)[0]))
        
        return absorption,tau,contrib_absorption
        

    def compute_absorption(self,tau,dz):
        import numexpr as ne
        tau = ne.evaluate('exp(-tau)')
        ap = self.altitudeProfile[:,None]
        pradius = self._planet.radius
        sradius = self._star.radius
        _dz = dz[:,None]
        #integral = (self._planet.radius+self.altitudeProfile[:,None])*(1.0-tau)*dz[:,None]
        #integral*=2.0
        #integral = np.sum(integral,axis=0)
        integral = ne.evaluate('sum((pradius+ap)*(1.0-tau)*_dz*2,axis=0)')
        return ((self._planet.radius**2.0) + integral)/(self._star.radius**2),tau





        
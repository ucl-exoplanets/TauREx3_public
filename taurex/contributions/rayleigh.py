 
from .contribution import Contribution,contribute_tau
import numpy as np
import numba



class RayleighContribution(Contribution):


    def __init__(self):
        super().__init__('Rayleigh')
 

    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):

        contrib = contribute_tau(start_horz_layer,end_horz_layer,
            density_offset,self.sigma_rayleigh,density,path_length,self._nlayers,self._ngrid,layer)
        
        self._total_contrib[layer] += contrib
        
        return contrib

    def build(self,model):
        pass
    

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib
 

    def prepare_each(self,model,wngrid):
        from taurex.util.scattering import rayleigh_sigma_from_name

        self._ngrid = wngrid.shape[0]
        self._nmols = 1
        self._nlayers = model.nLayers
        molecules = model.chemistry.activeGases + model.chemistry.inactiveGases

        for gasname in molecules:
            if np.sum(model.chemistry.get_gas_mix_profile(gasname)) == 0.0:
                continue
            sigma = rayleigh_sigma_from_name(gasname,wngrid)
            if sigma is not None:
                yield gasname,sigma[None,:]*model.chemistry.get_gas_mix_profile(gasname)[:,None]
    
    def prepare(self,model,wngrid):
        
        # precalculate rayleigh scattering cross sections

        self.info('Compute Rayleigh scattering cross sections')


        sigma_rayleigh = [x for x in self.prepare_each(model,wngrid)]


        self._nmols = len(sigma_rayleigh)


        self._nlayers = model.nLayers
        
        self.sigma_rayleigh = sum([x[1] for x in sigma_rayleigh])

        self.debug('Final sigma %s',self.sigma_rayleigh)
        self.info('Computing Ray interpolation ')
        
        self.info('DONE!!!')
        self._total_contrib = np.zeros(shape=(model.nLayers,wngrid.shape[0],))



    @property
    def sigma(self):
        return self.sigma_rayleigh
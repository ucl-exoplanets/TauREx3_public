 
from .contribution import Contribution,contribute_tau
import numpy as np
import numba



class RayleighContribution(Contribution):


    def __init__(self):
        super().__init__('Rayleigh')

    def build(self,model):
        pass
    

    def finalize(self,model):
        raise NotImplementedError

 

    def prepare_each(self,model,wngrid):
        from taurex.util.scattering import rayleigh_sigma_from_name

        #self._total_contrib = np.zeros(shape=(model.nLayers,wngrid.shape[0],))
        self._ngrid = wngrid.shape[0]
        self._nmols = 1
        self._nlayers = model.nLayers
        molecules = model.chemistry.activeGases + model.chemistry.inactiveGases
        
        for gasname in molecules:
            #self._total_contrib[...] =0.0
            if np.max(model.chemistry.get_gas_mix_profile(gasname)) == 0.0:
                continue
            sigma = rayleigh_sigma_from_name(gasname,wngrid)
            
            if sigma is not None:
                final_sigma = sigma[None,:]*model.chemistry.get_gas_mix_profile(gasname)[:,None]
                self.sigma_xsec = final_sigma
                yield gasname,final_sigma
    

from .contribution import Contribution
import numpy as np
import numba

@numba.jit(nopython=True, nogil=True,parallel=True)
def absorption_numba(sigma,density,path,nlayers,ngrid,nmols,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(nlayers-layer):
        _path = path[k]
        _density = density[k+layer]
        for mol in range(nmols):
            for wn in numba.prange(ngrid):
                tau[wn] += sigma[k+layer,mol,wn]*_path*_density
    return tau


class AbsorptionContribution(Contribution):


    def __init__(self):
        super().__init__('Absorption')

    

    def contribute(self,model,layer,density,path_length,return_contrib):
        # sigma=self.sigma_xsec[layer:total_layers]
        # combined_pt_dt = (density*path_length)[...,None,None]

        # comp = ne.evaluate('sum(sigma*combined_pt_dt,axis=0)')
        # comp = ne.evaluate('sum(comp,axis=0)')
        self._total_contrib[layer] +=absorption_numba(self.sigma_xsec,density,path_length,self._nlayers,self._ngrid,self._nmols,layer)
        #if return_contrib:
        #self._total_contrib[layer] += comp
        #elf.debug('Contribution {}'.format(comp))
        #return comp

    def build(self,model):
        pass
        
    
    def prepare(self,model,wngrid):
        import numexpr as ne
        ngases = len(model._gas_profile.activeGases)
        self.debug('Creating crossection for wngrid {} with ngases {} and nlayers {}'.format(wngrid,ngases,model.nLayers))

        sigma_xsec = np.zeros(shape=(model.pressure_profile.nLayers,ngases,wngrid.shape[0]))
        


        for idx_gas,gas in enumerate(model._gas_profile.activeGases):

            self.info('Recomputing active gas {} opacity'.format(gas))
            for idx_layer,tp in enumerate(zip(model.temperatureProfile,model.pressureProfile)):
                self.debug('Got index,tp {} {}'.format(idx_layer,tp))
                temperature,pressure = tp
                pressure/=1e5
                sigma_xsec[idx_layer,idx_gas] = model.opacity_dict[gas].opacity(temperature,pressure,wngrid)
        

        active_gas = model._gas_profile.activeGasMixProfile.transpose()[...,None]
        
        self._ngrid = wngrid.shape[0]
        self._nlayers = model.pressure_profile.nLayers
        self._nmols = ngases

        self.sigma_xsec= ne.evaluate('sigma_xsec*active_gas')
        self.info('Done')
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))
        return self.sigma_xsec

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib

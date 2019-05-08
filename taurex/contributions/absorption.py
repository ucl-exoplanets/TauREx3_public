
from .contribution import Contribution
import numpy as np
import numba
from taurex.cache import OpacityCache
@numba.jit(nopython=True, nogil=True,parallel=True)
def absorption_numba(startK,endK,density_offset,sigma,density,path,nlayers,ngrid,nmols,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(startK,endK):
        _path = path[k]
        _density = density[k+density_offset]
        for mol in range(nmols):
            for wn in numba.prange(ngrid):
                tau[wn] += sigma[k+layer,mol,wn]*_path*_density
    return tau


class AbsorptionContribution(Contribution):


    def __init__(self):
        super().__init__('Absorption')
        self._opacity_cache = OpacityCache()
    

    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):
        contrib =absorption_numba(start_horz_layer,end_horz_layer,density_offset,self.sigma_xsec,density,path_length,self._nlayers,self._ngrid,self._nmols,layer)
        self._total_contrib[layer] += contrib
        return contrib

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
                sigma_xsec[idx_layer,idx_gas] = self._opacity_cache[gas].opacity(temperature,pressure,wngrid)
                self.debug('Sigma for T {}, P:{} is {}'.format(temperature,pressure,sigma_xsec[idx_layer,idx_gas]))

        active_gas = model._gas_profile.activeGasMixProfile.transpose()[...,None]
        

        self.debug('Sigma is {}'.format(sigma_xsec))
        self.debug('Active gas mix ratio is {}'.format(active_gas))


        self._ngrid = wngrid.shape[0]
        self._nlayers = model.pressure_profile.nLayers
        self._nmols = ngases

        self.sigma_xsec= sigma_xsec*active_gas


        self.debug('Final sigma is {}'.format(self.sigma_xsec))
        #quit()
        self.info('Done')
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))
        return self.sigma_xsec

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib

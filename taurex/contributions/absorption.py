from .contribution import Contribution,contribute_tau
import numpy as np
import numba
from taurex.cache import OpacityCache, GlobalCache
from taurex.cache.ktablecache import KTableCache
import math


@numba.jit(nopython=True, nogil=True,fastmath=True)
def contribute_ktau(startK,endK,density_offset,sigma,density,path,weights,tau,ngrid,layer,ngauss):
    tau_temp = np.zeros(shape=(ngrid,ngauss))
    
    for k in range(startK, endK):
        _path = path[k]
        _density = density[k+density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            for g in range(ngauss):
                tau_temp[wn,g] += sigma[k+layer,wn,g]*_path*_density

    for wn in range(ngrid):   
        transtemp = 0.0
        for g in range(ngauss):
            transtemp += math.exp(-tau_temp[wn,g])*weights[g]
        tau[layer,wn]+=-math.log(transtemp)


class AbsorptionContribution(Contribution):


    def contribute(self, model, start_horz_layer, end_horz_layer,
                   density_offset, layer, density, tau, path_length=None):
        
        if self._use_ktables:
            #startK,endK,density_offset,sigma,density,path,weights,tau,ngrid,layer,ngauss

            contribute_ktau(start_horz_layer, end_horz_layer, density_offset,
                            self.sigma_xsec, density, path_length,
                            self.weights, tau, self._ngrid, layer, 
                            self.weights.shape[0])
        else:
            super().contribute(model, start_horz_layer, end_horz_layer,
                   density_offset, layer, density, tau, path_length)

    def __init__(self):
        super().__init__('Absorption')
        self._opacity_cache = OpacityCache()
        

    def build(self,model):
        pass

    def prepare_each(self,model,wngrid):
        self.debug('Preparing model with %s', wngrid.shape)
        self._ngrid = wngrid.shape[0]
        self._use_ktables = GlobalCache()['opacity_method'] == 'ktables'
        self.info('Using cross-sections? %s', not self._use_ktables)
        weights = None

        if self._use_ktables:
            self._opacity_cache = KTableCache()
        else:
            self._opacity_cache = OpacityCache()
        sigma_xsec = None
        self.weights = None

        for gas in model.chemistry.activeGases:

            #self._total_contrib[...] =0.0
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info('Recomputing active gas %s opacity',gas)

            xsec = self._opacity_cache[gas]

            if self._use_ktables and self.weights is None:
                self.weights = xsec.weights
            
            if sigma_xsec is None:

                if self._use_ktables:
                    sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid, len(self.weights)))
                else:
                    sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid))
            else:
                sigma_xsec[...] = 0.0

                

            

            for idx_layer, tp in enumerate(zip(model.temperatureProfile, model.pressureProfile)):
                self.debug('Got index,tp %s %s', idx_layer,tp)
                
                temperature, pressure = tp
                #print(gas,self._opacity_cache[gas].opacity(temperature,pressure,wngrid),gas_mix[idx_layer])
                sigma_xsec[idx_layer] += xsec.opacity(temperature, pressure, wngrid)*gas_mix[idx_layer]

            self.sigma_xsec = sigma_xsec

            self.debug('SIGMAXSEC %s',self.sigma_xsec)
            
            yield gas, sigma_xsec
        
    def prepare(self, model, wngrid):
        """

        Used to prepare the contribution for the calculation.
        Called before the forward model performs the main optical depth
        calculation. Default behaviour is to loop through :func:`prepare_each`
        and sum all results into a single cross-section.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid
        """

        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = None
        self.debug('ABSORPTION VERSION')
        for gas, sigma in self.prepare_each(model, wngrid):
            self.debug('Gas %s', gas)
            self.debug('Sigma %s', sigma)
            if sigma_xsec is None:
                sigma_xsec = np.zeros_like(sigma)
            sigma_xsec += sigma

        self.sigma_xsec = sigma_xsec
        self.debug('Final sigma is %s', self.sigma_xsec)
        self.info('Done')

    


    def finalize(self,model):
        raise NotImplementedError



    @property
    def sigma(self):
        return self.sigma_xsec

    @classmethod
    def input_keywords(self):
        return ['Absorption', 'Molecules', ]
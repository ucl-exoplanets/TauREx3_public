
from .contribution import Contribution,contribute_tau
import numpy as np
import numba
from taurex.cache import OpacityCache




class AbsorptionContribution(Contribution):


    def __init__(self):
        super().__init__('Absorption')
        self._opacity_cache = OpacityCache()
    

    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):
        #print(self.sigma_xsec.max())
        contrib =contribute_tau(start_horz_layer,end_horz_layer,density_offset,self.sigma_xsec,density,path_length,self._nlayers,self._ngrid,layer)
        self._total_contrib[layer] += contrib
        return contrib

    def build(self,model):
        pass

    def prepare_each(self,model,wngrid):
        from taurex.util.scattering import rayleigh_sigma_from_name
        self._total_contrib = np.zeros(shape=(model.nLayers,wngrid.shape[0],))
        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = np.zeros(shape=(model.nLayers,wngrid.shape[0]))

        self._opacity_cache = OpacityCache()
        for gas in model.chemistry.activeGases:
            sigma_xsec[...]=0.0
            self._total_contrib[...] =0.0
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info('Recomputing active gas %s opacity',gas)

            xsec = self._opacity_cache[gas]
            for idx_layer,tp in enumerate(zip(model.temperatureProfile,model.pressureProfile)):
                self.debug('Got index,tp %s %s',idx_layer,tp)
                
                temperature,pressure = tp
                #print(gas,self._opacity_cache[gas].opacity(temperature,pressure,wngrid),gas_mix[idx_layer])
                sigma_xsec[idx_layer] += xsec.opacity(temperature,pressure,wngrid)*gas_mix[idx_layer]
            self.sigma_xsec= sigma_xsec
            yield gas,sigma_xsec
        


    
    def prepare(self,model,wngrid):
        import numexpr as ne
        ngases = len(model.chemistry.activeGases)
        self.debug('Creating crossection for wngrid %s with ngases %s and nlayers %s',wngrid,ngases,model.nLayers)




        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers
        self._nmols = ngases
        
        #print([x for x in self.prepare_each(model,wngrid)])

        sigma_xsec = np.zeros(shape=(self._nlayers,self._ngrid))

        for gas,sigma in self.prepare_each(model,wngrid):
            self.debug('Gas %s',gas)
            self.debug('Sigma %s',sigma)
            sigma_xsec += sigma

        self.sigma_xsec = sigma_xsec
        #print('FINAL SIGMA',self.sigma_xsec)

        self.debug('Final sigma is %s',self.sigma_xsec)
        #quit()
        self.info('Done')

        return self.sigma_xsec

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib


    @property
    def sigma(self):
        return self.sigma_xsec
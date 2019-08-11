
from .contribution import Contribution,contribute_tau
import numpy as np
import numba
from taurex.cache import OpacityCache




class AbsorptionContribution(Contribution):


    def __init__(self):
        super().__init__('Absorption')
        self._opacity_cache = OpacityCache()
    

    def build(self,model):
        pass

    def prepare_each(self,model,wngrid):
        from taurex.util.scattering import rayleigh_sigma_from_name


        sigma_xsec = np.zeros(shape=(model.nLayers,wngrid.shape[0]))

        self._opacity_cache = OpacityCache()
        for gas in model.chemistry.activeGases:
            sigma_xsec[...]=0.0
            #self._total_contrib[...] =0.0
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
        


    


    def finalize(self,model):
        raise NotImplementedError



    @property
    def sigma(self):
        return self.sigma_xsec
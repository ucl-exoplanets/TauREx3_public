
from .contribution import Contribution
import numpy as np

class AbsorptionContribution(Contribution):


    def __init__(self):
        super().__init__('Absorption')

    

    def contribute(self,model,layer,density,path_length):
        import numexpr as ne
        total_layers = model.pressure_profile.nLayers
        sigma=self.sigma_xsec[layer:total_layers]

        comp = ne.evaluate('sum(sigma*density*path_length,axis=0)')
        comp = ne.evaluate('sum(comp,axis=0)')
        self._total_contrib[layer] += comp
        return comp

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
        
        self.sigma_xsec= ne.evaluate('sigma_xsec*active_gas')
        self.info('Done')
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))
        return self.sigma_xsec

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib

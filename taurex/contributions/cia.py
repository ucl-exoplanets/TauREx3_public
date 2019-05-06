from .contribution import Contribution
import numpy as np
import numba

@numba.jit(nopython=True,parallel=True)
def cia_numba(sigma,density,path,nlayers,ngrid,nmols,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(nlayers-layer):
        _path = path[k]
        _density = density[k+layer]
        for mol in range(nmols):
            for wn in range(ngrid):
                tau[wn] += sigma[k+layer,mol,wn]*_path*_density*_density
    return tau


class CIAContribution(Contribution):


    def __init__(self,cia_pairs):
        super().__init__('CIA')
        self._cia_pairs = []


    @property
    def ciaPairs(self):
        return self._cia_pairs

    @ciaPairs.setter
    def ciaPairs(self,value):
        self._cia_pairs = value


    def contribute(self,model,layer,density,path_length,return_contrib):
        import numexpr as ne
        total_layers = model.pressure_profile.nLayers
        # sigma = self.sigma_cia[layer:total_layers,:]
        # combined_pt_dt = (density*density*path_length)[...,None,None]
        # contrib = ne.evaluate('sum(sigma*combined_pt_dt,axis=0)')

        # contrib = ne.evaluate('sum(contrib,axis=0)')
        if self._total_cia > 0:
            self._total_contrib[layer,:]+=cia_numba(self.sigma_cia,density,path_length,self._nlayers,self._ngrid,self._total_cia,layer)

        #if return_contrib:
        #self._total_contrib[layer,:]+=contrib
        #return contrib



    def build(self,model):
        pass
    
    def prepare(self,model,wngrid):
        self._total_cia = len(self.ciaPairs)
        total_cia = self._total_cia
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))
        if self._total_cia == 0:
            return
        self.sigma_cia = np.zeros(shape=(model.pressure_profile.nLayers,total_cia,wngrid.shape[0]))

        self._total_cia = total_cia
        self._nlayers = model.pressure_profile.nLayers
        self._ngrid = wngrid.shape[0]
        self.info('Computing CIA ')
        for cia_idx,cia in enumerate(self.ciaPairs):
            for idx_layer,temperature in enumerate(model.temperatureProfile):
                _cia_xsec = cia.cia(temperature,wngrid)
                cia_factor = model._gas_profile.get_gas_mix_profile(cia.pairOne)*model._gas_profile.get_gas_mix_profile(cia.pairTwo)

                self.sigma_cia[idx_layer,cia_idx] = _cia_xsec*cia_factor[idx_layer]

        

    @property
    def totalContribution(self):
        return self._total_contrib


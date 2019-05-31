from .contribution import Contribution,contribute_tau
import numpy as np
import numba
from taurex.cache import CIACache

@numba.jit(nopython=True, nogil=True)
def contribute_cia(startK,endK,density_offset,sigma,density,path,nlayers,ngrid,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(startK,endK):
        _path = path[k]
        _density = density[k+density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            tau[wn] += sigma[k+layer,wn]*_path*_density*_density
    return tau

class CIAContribution(Contribution):


    def __init__(self,cia_pairs=None):
        super().__init__('CIA')
        self._cia_pairs = cia_pairs

        self._cia_cache = CIACache()
        if self._cia_pairs is None:
            self._cia_pairs=[]


    @property
    def ciaPairs(self):
        return self._cia_pairs

    @ciaPairs.setter
    def ciaPairs(self,value):
        self._cia_pairs = value


    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):
        if self._total_cia > 0:
            contrib =contribute_cia(start_horz_layer,end_horz_layer,density_offset,self.sigma_cia,density,path_length,self._nlayers,self._ngrid,layer)
            self._total_contrib[layer] += contrib
            return contrib
        else:
            return 0.0
        #if self._total_cia > 0:
            #self._total_contrib[layer,:]+=cia_numba(self.sigma_cia,density,path_length,self._nlayers,self._ngrid,self._total_cia,layer)



    def build(self,model):
        pass
    
    def prepare(self,model,wngrid):
        self._total_cia = len(self.ciaPairs)
        total_cia = self._total_cia
        self._total_contrib = np.zeros(shape=(model.nLayers,wngrid.shape[0],))
        if self._total_cia == 0:
            return
        self.sigma_cia = np.zeros(shape=(model.nLayers,total_cia,wngrid.shape[0]))

        self._total_cia = total_cia
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        self.info('Computing CIA ')
        for cia_idx,pairName in enumerate(self.ciaPairs):
            for idx_layer,temperature in enumerate(model.temperatureProfile):
                cia = self._cia_cache[pairName]

                _cia_xsec = cia.cia(temperature,wngrid)
                cia_factor = model.chemistry.get_gas_mix_profile(cia.pairOne)*model.chemistry.get_gas_mix_profile(cia.pairTwo)

                self.sigma_cia[idx_layer,cia_idx] = _cia_xsec*cia_factor[idx_layer]

        self.sigma_cia = np.sum(self.sigma_cia,axis=1)
        

    @property
    def totalContribution(self):
        return self._total_contrib


    def write(self,output):
        contrib = super().write(output)
        contrib.write_string_array('cia-pairs',self.ciaPairs)
        return contrib



    @property
    def sigma(self):
        return self.sigma_cia
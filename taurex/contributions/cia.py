from .contribution import Contribution,contribute_tau
import numpy as np
import numba
from taurex.cache import CIACache

@numba.jit(nopython=True, nogil=True)
def contribute_cia(startK,endK,density_offset,sigma,density,path,nlayers,ngrid,layer,tau):
    for k in range(startK,endK):
        _path = path[k]
        _density = density[k+density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            tau[layer,wn] += sigma[k+layer,wn]*_path*_density*_density

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


    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,tau,path_length=None):
        if self._total_cia > 0:
            contribute_cia(start_horz_layer,end_horz_layer,density_offset,self.sigma_xsec,density,path_length,self._nlayers,self._ngrid,layer,tau)
            #self._total_contrib[layer] += contrib
        #if self._total_cia > 0:
            #self._total_contrib[layer,:]+=cia_numba(self.sigma_cia,density,path_length,self._nlayers,self._ngrid,self._total_cia,layer)



    def build(self,model):
        pass
    

    def prepare_each(self,model,wngrid):

        self._total_cia = len(self.ciaPairs)
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        self.info('Computing CIA ')

        sigma_cia = np.zeros(shape=(model.nLayers,wngrid.shape[0]))

        #self._total_contrib = np.zeros(shape=(model.nLayers,wngrid.shape[0],))


        for pairName in self.ciaPairs:
            cia = self._cia_cache[pairName]
            sigma_cia[...]=0.0
            #self._total_contrib[...] =0.0
            cia_factor = model.chemistry.get_gas_mix_profile(cia.pairOne)*model.chemistry.get_gas_mix_profile(cia.pairTwo)

            last_temp = -1.0
            last_sigma = None

            for idx_layer,temperature in enumerate(model.temperatureProfile):
                _cia_xsec = None
                if last_temp == temperature:
                    _cia_xsec = last_sigma
                else:
                    _cia_xsec = cia.cia(temperature,wngrid)
                    last_temp = temperature
                    last_sigma = _cia_xsec
                sigma_cia[idx_layer] += _cia_xsec*cia_factor[idx_layer]
            self.sigma_xsec = sigma_cia
            yield pairName,sigma_cia

        

    @property
    def totalContribution(self):
        return self._total_contrib


    def write(self,output):
        contrib = super().write(output)
        contrib.write_string_array('cia_pairs',self.ciaPairs)
        return contrib

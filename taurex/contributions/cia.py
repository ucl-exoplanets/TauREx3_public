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
    

    def prepare_each(self,model,wngrid):

        self._total_cia = len(self.ciaPairs)
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        self.info('Computing CIA ')

        sigma_cia = np.zeros(shape=(model.nLayers,wngrid.shape[0]))

        self._total_contrib = np.zeros(shape=(model.nLayers,wngrid.shape[0],))


        for pairName in self.ciaPairs:
            cia = self._cia_cache[pairName]
            sigma_cia[...]=0.0
            self._total_contrib[...] =0.0
            cia_factor = model.chemistry.get_gas_mix_profile(cia.pairOne)*model.chemistry.get_gas_mix_profile(cia.pairTwo)
            for idx_layer,temperature in enumerate(model.temperatureProfile):

                
                _cia_xsec = cia.cia(temperature,wngrid)
                sigma_cia[idx_layer] += _cia_xsec*cia_factor[idx_layer]
            self.sigma_cia = sigma_cia
            yield pairName,sigma_cia

    def prepare(self,model,wngrid):
        

        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        self.info('Computing CIA ')

        sigma_cia = [x for x in self.prepare_each(model,wngrid)]


        if len(sigma_cia) ==0:
            return 
        sigma_cia = np.zeros(shape=(self._nlayers,self._ngrid))

        for gas,sigma in self.prepare_each(model,wngrid):
            self.debug('Gas %s',gas)
            self.debug('Sigma %s',sigma)
            sigma_cia += sigma

        self.sigma_cia = sigma_cia

        

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
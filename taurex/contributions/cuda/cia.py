
import numpy as np
from taurex.contributions import CIAContribution
from numba import cuda

@cuda.jit
def cia_cuda(sigma,density,path,nlayers,ngrid,nmols,layer,tau):

    wn = cuda.grid(1)
    result = 0.0

    if wn < ngrid:
        total_layers =nlayers-layer
        for k in range(total_layers):
            _path =path[k]
            _density = density[k+layer]
            for mol in range(nmols):
                #for wn in range(startY,ngrid,gridY):
                result += sigma[k+layer,mol,wn]*_path*_density*_density

        tau[layer,wn] += result




class GPUCIAContribution(CIAContribution):


    def __init__(self,cia_pairs=None):
        super().__init__(cia_pairs=cia_pairs)
        self._stream = cuda.stream()

    def contribute(self,model,layer,density,path_length,return_contrib):
        # sigma=self.sigma_xsec[layer:total_layers]
        # combined_pt_dt = (density*path_length)[...,None,None]

        # comp = ne.evaluate('sum(sigma*combined_pt_dt,axis=0)')
        # comp = ne.evaluate('sum(comp,axis=0)')


        self.cache_path(model.path_length)
        cia_cuda[self._griddim,self._blockdim,self._stream](self._device_sigma,self._device_dens,self._dpath[layer],self._nlayers,self._ngrid,self._total_cia,layer,self._device_tau)

    def cache_path(self,path):

        if self._dpath is None:
            self._dpath =[]
            for p in path:
                self._dpath.append(cuda.to_device(p,stream=self._stream))

    @property
    def totalContribution(self):
        self._device_tau.to_host(stream=self._stream)
        self._stream.synchronize()
        return self._total_contrib

    def prepare(self,model,wngrid):
        super().prepare(model,wngrid)

        self.info('Transfering grid to GPU')

        self._device_tau = cuda.to_device(self._total_contrib,stream=self._stream)

        self._device_dens = cuda.to_device(model.densityProfile,stream=self._stream)
        self._device_sigma = cuda.to_device(self.sigma_cia,stream=self._stream)
        self.info('Transfer complete')

        self._griddim = 1024
        self._blockdim = (wngrid.shape[0] + (self._griddim - 1)) // self._griddim
        self._dpath = None


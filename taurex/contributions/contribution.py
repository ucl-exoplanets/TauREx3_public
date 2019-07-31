"""
Base contribution classes and functions for computing optical depth
"""

from taurex.log import Logger
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.output.writeable import Writeable
import numba

@numba.jit(nopython=True, nogil=True)
def contribute_tau(startK,endK,density_offset,sigma,density,path,nlayers,ngrid,layer):
    """
    Generic contribution function for tau, numba-fied for performance.

    Parameters
    ----------
    startK : int
        starting horizontal layer index of the atmosphere to loop
    
    endK : int
        ending horizontal layer index of the atmosphere to loop
    
    density_offset : int
        Which part of the density profile to start from
    
    sigma : array_like
        cross-section to compute contribution
    
    density : array_like
        density profile of atmosphere
    
    path : array_like
        path-length or altitude gradient
    
    nlayers : int
        Total number of layers (unused)
    
    ngrid : int
        total number of grid points 
    
    layer : int
        Which layer we currently on

    
    Returns
    -------
    tau : array_like 
        optical depth (well almost you still need to do ``exp(-tau)`` yourself )

    """
    tau = np.zeros(shape=(ngrid,))
    for k in range(startK,endK):
        _path = path[k]
        _density = density[k+density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            tau[wn] += sigma[k+layer,wn]*_path*_density
    return tau


class Contribution(Fittable,Logger,Writeable):



    def __init__(self,name):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        self._name = name
        self._total_contribution = None
        self._enabled = True
    @property
    def name(self):
        return self._name


  

    def contribute(self,model,layer,density,start_horz_layer,end_horz_layer,density_offset,path_length=None):
        raise NotImplementedError

    def build(self,model):
        raise NotImplementedError
    
    def prepare(self,model,wngrid):
        raise NotImplementedError

    def finalize(self,model):
        raise NotImplementedError


    @property
    def sigma(self):
        raise NotImplementedError

    @property
    def totalContribution(self):
        raise NotImplementedError

    

    def write(self,output):
        contrib = output.create_group(self.name)
        return contrib


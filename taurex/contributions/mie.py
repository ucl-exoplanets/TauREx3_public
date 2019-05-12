 
from .contribution import Contribution
import numpy as np
import numba
import math
from taurex.data.fittable import fitparam
from taurex.external.mie import bh_mie
@numba.jit(nopython=True,parallel=True, nogil=True)
def mie_numba(startK,endK,density_offset,sigma,density,path,nlayers,ngrid,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(startK,endK):
        _path = path[k]
        _density = density[k+density_offset]
        for wn in range(ngrid):
            tau[wn] += sigma[wn]*_path*_density

    return tau


class MieContribution(Contribution):


    def __init__(self,mie_path=None,mie_type='cloud'):
        super().__init__('Mie')
        self._mie_path = None
        self.load_mie_indices()
        self._mie_type = mie_type.lower()
    def load_mie_indices(self):
        import pathlib
        if self._mie_path is None:
            raise Exception('No mie file path defined')
        #loading file 
        mie_raw = np.loadtxt(self._mie_path,skiprows=1)

        #saving to memory
        species_name = pathlib.Path(self._mie_path).stem
        self.info('Preloading Mie refractive indices for %s' % species_name)
        self.mie_indices = mie_raw
        self.mie_species = species_name       

        self._mie_radius = 1.0
        self._mix_cloud_mix = 1.0
        self._cloud_top_pressure = 1.0
        self._cloud_bottom_pressure = -1.0


    @property
    def mieSpecies(self):
        return self.mie_species

    @property
    def wavelengthGrid(self):
        return self.mie_indices[:,0]
    
    @property
    def wavenumberGrid(self):
        return 10000/self.wavelengthGrid

    @property
    def realReference(self):
        return self.mie_indices[:,1]
    
    @property
    def imaginaryReference(self):
        return self.mie_indices[:,2]

    @property
    def mieType(self):
        return self._mie_type

    @fitparam(param_name='log_clouds_particle_size',param_latex='log($R_\\mathrm{clouds}$)',default_fit=False,default_bounds=[-10,1])
    def particleSize(self):
        return math.log10(self._mie_radius)
    
    @particleSize.setter
    def particleSize(self,value):
        self._mie_radius = 10.0**value
    

    @fitparam(param_name='log_clouds_topP',param_latex='$log(P_\\mathrm\{top\})$',default_fit=False,default_bounds=[-1,1])
    def cloudTopPressure(self):
        return math.log10(self._cloud_top_pressure)
    
    @cloudTopPressure.setter
    def cloudTopPressure(self,value):
        self._cloud_top_pressure = 10**value

    @fitparam(param_name='log_clouds_bottomP',param_latex='$log(P_\\mathrm\{bottom\})$',default_fit=False,default_bounds=[-1,1])
    def cloudBottomPressure(self):
        return math.log10(self._cloud_bottom_pressure)
    
    @cloudBottomPressure.setter
    def cloudBottomPressure(self,value):
        self._cloud_bottom_pressure = 10**value

    @fitparam(param_name='log_cloud_mixing',param_latex='log($\\chi_\\mathrm\{clouds\}$)',default_fit=False,default_bounds=[-1,1])
    def cloudMixing(self):
        return math.log10(self._mix_cloud_mix)
    
    @cloudMixing.setter
    def cloudMixing(self,value):
        self._mix_cloud_mix = 10**value


    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):

        contrib = mie_numba(start_horz_layer,end_horz_layer,
            density_offset,self.sigma_mie,density,path_length,self._nlayers,self._ngrid,layer)
        self._total_contrib[layer,:]+=contrib
        return contrib

    def build(self,model):
        wavegrid = self.wavelengthGrid*1e-4 #micron to cm
        a = self._mie_radius * 1e-4 #micron to cm
#         agrid *= 1e-4 #micron to cm
        agrid = None
        na= None
        #getting particle size distribution
        if self.mieType == 'cloud':
            agrid = np.linspace(1e-7,a*3,30) #particle size distribution micron grid
            na = (agrid/a)**6 * np.exp((-6.0*(agrid/a)))  #earth clouds equ. 36 Sharp & Burrows 2007
        elif self.mieType == 'haze':
            agrid = np.linspace(1e-7,a*15,50) #particle size distribution micron grid
            na = agrid/a * np.exp((-2.0*(agrid/a)**0.5)) #haze distributino equ. 37 Sharp & Burrows 2007
        else:
            raise Exception('Unknown Mie type {}'.format(self.mieType))
        na /= np.max(na) #normalise into weigtings
        na_clip = na[na>1e-3] #curtails wings
        agrid_clip = agrid[na>1e-3]


        #running Mie model for particle sizes in distribution
        sig_out = np.ndarray(shape=(len(wavegrid),len(agrid_clip)))
        for i,ai in enumerate(agrid_clip):

            sig_out[:,i] = bh_mie(ai,wavegrid,self.realReference,self.imaginaryReference)

        #average mie cross section weighted by particle size distribution
        self._sig_out_aver = np.average(sig_out,weights=na_clip,axis=1)
    

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib
 
    def prepare(self,model,wngrid):
        self._nlayer = model.nLayers
        self._ngrid = wngrid.shape[0]
        self.sigma_mie = np.interp(wngrid,self.wavenumberGrid,self._sig_out_aver)*self._mix_cloud_mix
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))
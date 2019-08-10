 
from .contribution import Contribution,contribute_tau
import numpy as np
import numba
import math
from taurex.data.fittable import fitparam
from taurex.external.mie import bh_mie



class MieContribution(Contribution):


    def __init__(self,mie_path=None,mie_type='cloud',particle_radius=1.0,cloud_mix=1.0,cloud_bottom_pressure=1e-2,
                    cloud_top_pressure=1e-3):
        super().__init__('Mie')
        self._mie_path = mie_path
        self.load_mie_indices()
        self._mie_type = mie_type.lower()

        self._mie_radius = 1.0
        self._mix_cloud_mix = 1.0
        self._cloud_top_pressure = 1.0
        self._cloud_bottom_pressure = -1.0
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
        return np.ascontiguousarray(self.mie_indices[:,1])
    
    @property
    def imaginaryReference(self):
        return np.ascontiguousarray(self.mie_indices[:,2])

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


    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,tau,path_length=None):
        if model.pressureProfile[layer] <= self._cloud_bottom_pressure and model.pressureProfile[layer] >= self._cloud_top_pressure :
            contrib = contribute_tau(start_horz_layer,end_horz_layer,
                density_offset,self.sigma_mie,density,path_length,self._nlayers,self._ngrid,layer)
            #self._total_contrib[layer,:]+=contrib
            return contrib
        else:
            return 0.0

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

    

    def prepare_each(self,model,wngrid):
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        self.sigma_mie = np.interp(wngrid,self.wavenumberGrid,self._sig_out_aver)*self._mix_cloud_mix
        #self._total_contrib[...]=0.0
        yield 'Mie',self.sigma_mie


    def write(self,output):
        contrib = super().write(output)
        contrib.write_scalar('cloud_particle_size',self._mie_radius)
        contrib.write_scalar('cloud_top_pressure',self._cloud_top_pressure)
        contrib.write_scalar('cloud_bottom_pressure',self._cloud_bottom_pressure)
        contrib.write_scalar('cloud_mixing',self._mix_cloud_mix)
        contrib.write_array('averaged_sigma',self._sig_out_aver)
        contrib.write_array('wavenumber_grid',self.wavenumberGrid)
        return contrib
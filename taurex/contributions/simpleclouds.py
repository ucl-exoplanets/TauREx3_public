 
from .contribution import Contribution
import numpy as np
from taurex.data.fittable import fitparam
import math

class SimpleCloudsContribution(Contribution):

    def __init__(self,clouds_pressure = 1e3):
        super().__init__('SimpleClouds')
        self._cloud_pressure = clouds_pressure

    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):

        if model.pressureProfile[layer] >= self._cloud_pressure:
            #Set to infinity so zero opacity
            self._total_contrib[layer,:]=np.inf
        return np.inf


    def build(self,model):
        pass
    
    def prepare(self,model,wngrid):
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))

    @fitparam(param_name='log_clouds_pressure',param_latex='$log(P_\mathrm{clouds})$',default_fit=False,default_bounds=[-3, 6])
    def cloudsPressure(self):
        return math.log10(self._cloud_pressure)
    
    @cloudsPressure.setter
    def cloudsPressure(self,value):
        self._cloud_pressure = 10**value

    @property
    def totalContribution(self):
        return self._total_contrib


    def write(self,output):
        contrib = super().write(output)
        contrib.write_string_array('cloud_pressure',self._cloud_pressure)
        return contrib
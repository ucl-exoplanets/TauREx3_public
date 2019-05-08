 
from .contribution import Contribution
import numpy as np



class SimpleCloudsContribution(Contribution):

    def __init__(self,clouds_pressure = 1e3):
        super().__init__('SimpleClouds')
        self._cloud_pressure = clouds_pressure

    def contribute(self,model,start_horz_layer,end_horz_layer,density_offset,layer,density,path_length=None):

        if model.pressureProfile[layer] >= self._cloud_pressure:
            self._total_contrib[layer,:]+=1e100     
    


    def build(self,model):
        pass
    
    def prepare(self,model,wngrid):
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))

        

    @property
    def totalContribution(self):
        return self._total_contrib
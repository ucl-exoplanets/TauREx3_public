from .twopointgas import TwoPointGas
from taurex.util import molecule_texlabel,movingaverage
import math
import numpy as np
class TwoLayerGas(TwoPointGas):
    """

    Two layer gas profile.

    A gas profile with two different mixing layers at the surface of the planet and
    top of the atmosphere seperated at a defined
    pressure point and smoothened.



    Parameters
    -----------
    molecule_name : str
        Name of molecule

    mix_ratio_surface : float
        Mixing ratio of the molecule on the planet surface

    mix_ratio_top : float
        Mixing ratio of the molecule at the top of the atmosphere

    mix_ratio_P : float
        Boundary Pressure point between the two layers 

    mix_ratio_smoothing : float , optional
        smoothing window 

    """


    def __init__(self,molecule_name='CH4',mix_ratio_surface=1e-4,mix_ratio_top=1e-8,
                    mix_ratio_P=1e3,mix_ratio_smoothing=10):
        super().__init__(molecule_name=molecule_name,mix_ratio_surface=mix_ratio_surface,
                        mix_ratio_top=mix_ratio_top)
        

        self._mix_ratio_pressure = mix_ratio_P
        self._mix_ratio_smoothing = mix_ratio_smoothing

        self.add_P_param()


    @property
    def mixRatioPressure(self):
        return self._mix_ratio_pressure
    
    @property
    def mixRatioSmoothing(self):
        return self._mix_ratio_smoothing

    
    @mixRatioPressure.setter
    def mixRatioPressure(self,value):
        self._mix_pressure = value
    
    @mixRatioSmoothing.setter
    def mixRatioSmoothing(self,value):
        self._mix_smoothing = value


    def add_P_param(self):
        mol_name = self.molecule
        mol_tex = molecule_texlabel(mol_name)

        param_P = '{}_P'.format(mol_name)
        param_P_tex = '{}_P'.format(mol_tex)

        def read_P(self):
            return self._mix_ratio_pressure
        def write_P(self,value):
            self._mix_ratio_pressure = value

        fget_P = read_P
        fset_P = write_P

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_P,param_P_tex ,fget_P,fset_P,'log',default_fit,bounds) 
    
    def initialize_profile(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        self._mix_profile = np.zeros(nlayers)
        
        smooth_window = self._mix_ratio_smoothing
        P_layer = np.abs(pressure_profile - self._mix_ratio_pressure).argmin()

        start_layer = max(int(P_layer-smooth_window/2),0)
        end_layer = min(int(P_layer+smooth_window/2),nlayers-1)

        Pnodes = [pressure_profile[0], pressure_profile[start_layer], pressure_profile[end_layer], pressure_profile[-1]]
        Cnodes = [self.mixRatioSurface, self.mixRatioSurface, self.mixRatioTop,self.mixRatioTop]

        
        chemprofile = 10**np.interp((np.log(pressure_profile[::-1])), np.log(Pnodes[::-1]), np.log10(Cnodes[::-1]))


        wsize = nlayers * (smooth_window / 100.0)
        if (wsize % 2 == 0):
            wsize += 1
        C_smooth = 10**movingaverage(np.log10(chemprofile), int(wsize))
        border = np.int((len(chemprofile) - len(C_smooth)) / 2)
        self._mix_profile =  chemprofile[::-1]
        self._mix_profile[border:-border] = C_smooth[::-1]

    def write(self,output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('mix_ratio_P',self.mixRatioPressure)
        gas_entry.write_scalar('mix_ratio_smoothing',self.mixRatioSmoothing)

        return gas_entry
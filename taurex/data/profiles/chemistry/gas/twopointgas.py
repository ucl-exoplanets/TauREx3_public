from .gas import Gas
from taurex.util import molecule_texlabel
import math
import numpy as np
class TwoPointGas(Gas):
    """

    Constant gas profile


    Parameters
    -----------


    """


    def __init__(self,molecule_name='CH4',mix_ratio_surface=1e-4,mix_ratio_top=1e-8):
        super().__init__('TwoPointGas',molecule_name)
        self._mix_surface = mix_ratio_surface
        self._mix_top = mix_ratio_top
        self.add_surface_param()
        self.add_top_param()
        self._mix_profile = None
    @property
    def mixProfile(self):
        return self._mix_profile

    @property
    def mixRatioSurface(self):
        return self._mix_surface
    
    @property
    def mixRatioTop(self):
        return self._mix_top

    
    @mixRatioSurface.setter
    def mixRatioSurface(self,value):
        self._mix_surface = value
    
    @mixRatioTop.setter
    def mixRatioTop(self,value):
        self._mix_top = value


    def add_surface_param(self):
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)
        
        param_surface = '{}_S'.format(param_name)
        param_surf_tex = '{}_S'.format(param_tex)

        def read_surf(self):
            return self._mix_surface
        def write_surf(self,value):
            self._mix_surface = value

        fget_surf = read_surf
        fset_surf = write_surf

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_surface,param_surf_tex ,fget_surf,fset_surf,'log',default_fit,bounds)   

    def add_top_param(self):
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)
        
        param_top = '{}_T'.format(param_name)
        param_top_tex = '{}_T'.format(param_tex)

        def read_top(self):
            return self._mix_top
        def write_top(self,value):
            self._mix_top = value

        fget_top = read_top
        fset_top = write_top

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_top,param_top_tex ,fget_top,fset_top,'log',default_fit,bounds) 

    def initialize_profile(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        self._mix_profile = np.zeros(nlayers)
        
        chem_surf = self._mix_surface
        chem_top = self._mix_top
        p_surf = pressure_profile[0]
        p_top = pressure_profile[-1]

        a = (math.log10(chem_surf)-math.log10(chem_top))/(math.log10(p_surf)-math.log10(p_top))
        b = math.log10(chem_surf)-a*math.log10(p_surf)

        self._mix_profile[1:-1] =  10**(a * np.log10(pressure_profile[1:-1])+b)
        self._mix_profile[0] = chem_surf
        self._mix_profile[-1] = chem_top
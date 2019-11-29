from .gas import Gas
from taurex.util import molecule_texlabel,movingaverage
import math
import numpy as np
class PowerGas(Gas):
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


    def __init__(self,molecule_name='H2O',mix_ratio_surface=False, alpha=False,
                    beta= False, gamma= False):
        super().__init__('PowerGas',molecule_name=molecule_name)

        self._mix_surface = mix_ratio_surface
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._mix_profile = None
        self.add_surface_param()
        self.add_alpha_param()
        self.add_beta_param()
        self.add_gamma_param()

    @property
    def mixProfile(self):
        return self._mix_profile

    @property
    def mixRatioSurface(self):
        """Abundance on the planets surface"""
        return self._mix_surface

    @property
    def mixRatioTop(self):
        """Abundance on the top of atmosphere"""
        return self._mix_top

    @property
    def mixRatioPressure(self):
        return self._mix_ratio_pressure
    
    @property
    def mixRatioSmoothing(self):
        return self._mix_ratio_smoothing

    @mixRatioSurface.setter
    def mixRatioSurface(self, value):
        self._mix_surface = value

    @mixRatioTop.setter
    def mixRatioTop(self, value):
        self._mix_top = value
    
    @mixRatioPressure.setter
    def mixRatioPressure(self,value):
        self._mix_pressure = value
    
    @mixRatioSmoothing.setter
    def mixRatioSmoothing(self,value):
        self._mix_smoothing = value

    def add_surface_param(self):
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_surface = '{}_surface'.format(param_name)
        param_surf_tex = '{}_surface'.format(param_tex)

        def read_surf(self):
            return self._mix_surface

        def write_surf(self, value):
            self._mix_surface = value

        fget_surf = read_surf
        fset_surf = write_surf

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_surface, param_surf_tex, fget_surf, fset_surf, 'log', default_fit, bounds)

    def add_alpha_param(self):
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_alpha = '{}_alpha'.format(param_name)
        param_alpha_tex = '{}_alpha'.format(param_tex)

        def read_alpha(self):
            return self._alpha

        def write_alpha(self, value):
            self._alpha = value

        fget_alpha = read_alpha
        fset_alpha = write_alpha

        bounds = [0.5, 2.5]

        default_fit = False
        self.add_fittable_param(param_alpha, param_alpha_tex,fget_alpha,fset_alpha,'linear',default_fit,bounds)

    def add_beta_param(self):
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_beta = '{}_beta'.format(param_name)
        param_beta_tex = '{}_beta'.format(param_tex)

        def read_beta(self):
            return self._beta

        def write_beta(self, value):
            self._beta = value

        fget_beta = read_beta
        fset_beta = write_beta

        bounds = [1e4, 6e4]

        default_fit = False
        self.add_fittable_param(param_beta, param_beta_tex,fget_beta,fset_beta,'log',default_fit,bounds)

    def add_gamma_param(self):
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_gamma = '{}_gamma'.format(param_name)
        param_gamma_tex = '{}_gamma'.format(param_tex)

        def read_gamma(self):
            return self._gamma

        def write_gamma(self, value):
            self._gamma = value

        fget_gamma = read_gamma
        fset_gamma = write_gamma

        bounds = [5, 25]

        default_fit = False
        self.add_fittable_param(param_gamma, param_gamma_tex,fget_gamma,fset_gamma,'linear',default_fit,bounds)

    def check_known(self, molecule_name ='H2O'):

        known_molecules = ['H2','H2O','TiO','VO', 'H-', 'Na', 'K']
        a = [1.,2.,1.6,1.5,0.6,0.6,0.6]
        b = [2.41,4.83,5.94,5.4,-0.14,1.89,1.28]
        g = [6.5,15.9,23.0,23.8,7.7,12.2,12.7]
        g *= 1e4
        A = [-0.1,-3.3,-7.1,-9.2,-8.3,-5.5,-7.1]
        A = np.power(10, A)
        if molecule_name in known_molecules:
            i = known_molecules.index()
            return a[i], b[i], g[i], A[i]
        else:
            return None, None, None, None


    
    def initialize_profile(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        self._mix_profile = np.zeros(nlayers)
        molecule_name = self._molecule_name
        coeffs = self.check_known(molecule_name=molecule_name)
        mix_surface = self._mix_surface
        alpha = self._alpha
        beta = self._beta
        gamma = self._gamma

        if self._mix_surface is None:
            if coeffs[3] is not None:
                mix_surface =coeffs[3]
            else:
                self.error('molecule %s has a missing power coefficient', molecule_name)
                raise ValueError
        if self._alpha is None:
            if coeffs[0] is not None:
                alpha = coeffs[0]
            else:
                self.error('molecule %s has a missing power coefficient', molecule_name)
                raise ValueError
        if self._beta is None:
            if coeffs[1] is not None:
                beta = coeffs[1]
            else:
                self.error('molecule %s has a missing power coefficient', molecule_name)
                raise ValueError
        if self._gamma is None:
            if coeffs[2] is not None:
                gamma = coeffs[2]
            else:
                self.error('molecule %s has a missing power coefficient', molecule_name)
                raise ValueError

        Ad = alpha*np.log10(pressure_profile)+beta/temperature_profile+gamma
        self._mix_profile = 1/(mix_surface**(-0.5)+Ad**(-0.5))**2


    def write(self, output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('mix_ratio_top', self.mixRatioTop)
        gas_entry.write_scalar('mix_ratio_surface', self.mixRatioSurface)
        gas_entry.write_scalar('mix_ratio_P',self.mixRatioPressure)
        gas_entry.write_scalar('mix_ratio_smoothing',self.mixRatioSmoothing)

        return gas_entry
from .gas import Gas
from taurex.util import movingaverage, molecule_texlabel
import numpy as np


class TwoLayerGas(Gas):
    """

    Two layer gas profile.

    A gas profile with two different mixing layers at the surface of the
    planet and top of the atmosphere seperated at a defined
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

    def __init__(self, molecule_name='CH4', mix_ratio_surface=1e-4,
                 mix_ratio_top=1e-8, mix_ratio_P=1e3,
                 mix_ratio_smoothing=10):
        super().__init__(self.__class__.__name__, molecule_name=molecule_name)

        self._mix_surface = mix_ratio_surface
        self._mix_top = mix_ratio_top
        self._mix_ratio_pressure = mix_ratio_P
        self._mix_ratio_smoothing = mix_ratio_smoothing
        self._mix_profile = None
        self.add_surface_param()
        self.add_top_param()
        self.add_P_param()

    @property
    def mixProfile(self):
        """

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer

        """
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
    def mixRatioPressure(self, value):
        self._mix_pressure = value

    @mixRatioSmoothing.setter
    def mixRatioSmoothing(self, value):
        self._mix_smoothing = value

    def add_surface_param(self):
        """
        Generates surface fitting parameters. Has the form
        ''Moleculename_surface'
        """
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
        self.add_fittable_param(param_surface, param_surf_tex, fget_surf,
                                fset_surf, 'log', default_fit, bounds)

    def add_top_param(self):
        """
        Generates TOA fitting parameters. Has the form:
        'Moleculename_top'
        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_top = '{}_top'.format(param_name)
        param_top_tex = '{}_top'.format(param_tex)

        def read_top(self):
            return self._mix_top

        def write_top(self, value):
            self._mix_top = value

        fget_top = read_top
        fset_top = write_top

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_top, param_top_tex, fget_top,
                                fset_top, 'log', default_fit, bounds)

    def add_P_param(self):
        """
        Generates pressure fitting parameter. Has the form
        'Moleculename_P'
        """
        mol_name = self.molecule
        mol_tex = molecule_texlabel(mol_name)

        param_P = '{}_P'.format(mol_name)
        param_P_tex = '{}_P'.format(mol_tex)

        def read_P(self):
            return self._mix_ratio_pressure

        def write_P(self, value):
            self._mix_ratio_pressure = value

        fget_P = read_P
        fset_P = write_P

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_P, param_P_tex,
                                fget_P, fset_P, 'log', default_fit, bounds)

    def initialize_profile(self, nlayers=None, temperature_profile=None,
                           pressure_profile=None, altitude_profile=None):
        self._mix_profile = np.zeros(nlayers)

        smooth_window = self._mix_ratio_smoothing
        P_layer = np.abs(pressure_profile - self._mix_ratio_pressure).argmin()

        start_layer = max(int(P_layer-smooth_window/2), 0)

        end_layer = min(int(P_layer+smooth_window/2), nlayers-1)

        Pnodes = [pressure_profile[0], pressure_profile[start_layer],
                  pressure_profile[end_layer], pressure_profile[-1]]

        Cnodes = [self.mixRatioSurface, self.mixRatioSurface,
                  self.mixRatioTop, self.mixRatioTop]

        chemprofile = 10**np.interp((np.log(pressure_profile[::-1])),
                                    np.log(Pnodes[::-1]),
                                    np.log10(Cnodes[::-1]))

        wsize = nlayers * (smooth_window / 100.0)

        if (wsize % 2 == 0):
            wsize += 1

        C_smooth = 10**movingaverage(np.log10(chemprofile), int(wsize))

        border = np.int((len(chemprofile) - len(C_smooth)) / 2)

        self._mix_profile = chemprofile[::-1]

        self._mix_profile[border:-border] = C_smooth[::-1]

    def write(self, output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('mix_ratio_top', self.mixRatioTop)
        gas_entry.write_scalar('mix_ratio_surface', self.mixRatioSurface)
        gas_entry.write_scalar('mix_ratio_P', self.mixRatioPressure)
        gas_entry.write_scalar('mix_ratio_smoothing', self.mixRatioSmoothing)

        return gas_entry

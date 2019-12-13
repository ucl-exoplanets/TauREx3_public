from .contribution import Contribution, contribute_tau
import numpy as np
from taurex.data.fittable import fitparam
from taurex.external.mie import bh_mie


class BHMieContribution(Contribution):
    """
    Computes a Mie scattering contribution using method given by
    Bohren & Huffman 2007

    Parameters
    ----------
    mie_path: str
        Path to molecular scattering parameters

    mie_type: str of either ``cloud`` or ``haze``
        Type of mie cloud

    bh_particle_radius: float
        Radius of scattering particles in um

    bh_clouds_mix: float
        Mix ratio in atmosphere

    bh_clouds_bottomP: float
        Bottom of cloud deck in Pa

    bh_clouds_topP: float
        Top of cloud deck in Pa


    """
    def __init__(self, mie_path=None, mie_type='cloud', bh_particle_radius=1.0,
                 bh_clouds_mix=1e-6, bh_clouds_bottomP=1e0,
                 bh_clouds_topP=1e-3):
        super().__init__('Mie')
        self._mie_path = mie_path
        self.load_mie_indices()
        self._mie_type = mie_type.lower()

        self._mie_radius = bh_particle_radius
        self._mix_cloud_mix = bh_cloud_mix
        self._cloud_top_pressure = bh_clouds_topP
        self._cloud_bottom_pressure = bh_clouds_bottomP

    def load_mie_indices(self):
        import pathlib
        if self._mie_path is None:
            raise Exception('No mie file path defined')
        # loading file
        mie_raw = np.loadtxt(self._mie_path, skiprows=1)

        # saving to memory
        species_name = pathlib.Path(self._mie_path).stem
        self.info('Preloading Mie refractive indices for %s' % species_name)
        self.mie_indices = mie_raw
        self.mie_species = species_name

    @property
    def mieSpecies(self):
        return self.mie_species

    @property
    def wavelengthGrid(self):
        return self.mie_indices[:, 0]

    @property
    def wavenumberGrid(self):
        return 10000/self.wavelengthGrid

    @property
    def realReference(self):
        return np.ascontiguousarray(self.mie_indices[:, 1])

    @property
    def imaginaryReference(self):
        return np.ascontiguousarray(self.mie_indices[:, 2])

    @property
    def mieType(self):
        return self._mie_type

    @fitparam(param_name='bh_particle_radius',
              param_latex='$R^{bh}_\mathrm{clouds}$',
              default_fit=False,
              default_bounds=[-10, 1])
    def particleSize(self):
        """
        Particle size in um
        """
        return self._mie_radius

    @particleSize.setter
    def particleSize(self, value):
        self._mie_radius = value

    @fitparam(param_name='bh_clouds_topP',
              param_latex='$P^{bh}_\mathrm{top}$',
              default_fit=False,
              default_bounds=[-1, 1])
    def cloudTopPressure(self):
        """
        Pressure at top of cloud deck in Pa
        """
        return self._cloud_top_pressure

    @cloudTopPressure.setter
    def cloudTopPressure(self, value):
        self._cloud_top_pressure = value

    @fitparam(param_name='bh_clouds_bottomP',
              param_latex='$P^{bh}_\mathrm{bottom}$',
              default_fit=False,
              default_bounds=[-1, 1])
    def cloudBottomPressure(self):
        """
        Pressure at bottom of cloud deck in Pa
        """
        return self._cloud_bottom_pressure

    @cloudBottomPressure.setter
    def cloudBottomPressure(self, value):
        self._cloud_bottom_pressure = value

    @fitparam(param_name='bh_clouds_mix',
              param_latex='$\chi^{bh}_\mathrm{clouds}$',
              default_fit=False,
              default_bounds=[-1, 1])
    def cloudMixing(self):
        """
        Cloud mixing ratio
        """
        return self._mix_cloud_mix

    @cloudMixing.setter
    def cloudMixing(self, value):
        self._mix_cloud_mix = value

    def contribute(self, model, start_layer, end_layer, density_offset,
                   layer, density, tau, path_length=None):
        """
        Contributes to optical depth between cloud pressures:
        ``bh_clouds_topP`` and ``bh_clouds_bottomP``

        """
        if model.pressureProfile[layer] <= self._cloud_bottom_pressure and \
                model.pressureProfile[layer] >= self._cloud_top_pressure:

            contribute_tau(start_layer, end_layer, density_offset,
                           self.sigma_mie, density, path_length, self._nlayers,
                           self._ngrid, layer, tau)

    def build(self, model):
        """
        Preperes mie scattering cross-section

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`

        """
        wavegrid = self.wavelengthGrid*1e-4  # micron to cm
        a = self._mie_radius * 1e-4  # micron to cm
        agrid = None
        na = None
        # getting particle size distribution
        if self.mieType == 'cloud':
            # particle size distribution micron grid
            agrid = np.linspace(1e-7, a*3, 30)
            # earth clouds equ. 36 Sharp & Burrows 2007
            na = (agrid/a)**6 * np.exp((-6.0*(agrid/a)))
        elif self.mieType == 'haze':
            # particle size distribution micron grid
            agrid = np.linspace(1e-7, a*15, 50)
            # haze distributino equ. 37 Sharp & Burrows 2007
            na = agrid/a * np.exp((-2.0*(agrid/a)**0.5))
        else:
            raise Exception('Unknown Mie type {}'.format(self.mieType))
        na /= np.max(na)  # normalise into weigtings
        na_clip = na[na > 1e-3]  # curtails wings
        agrid_clip = agrid[na > 1e-3]

        # Running Mie model for particle sizes in distribution
        sig_out = np.ndarray(shape=(len(wavegrid), len(agrid_clip)))
        for i, ai in enumerate(agrid_clip):

            sig_out[:, i] = bh_mie(ai, wavegrid, self.realReference,
                                   self.imaginaryReference)

        # average mie cross section weighted by particle size distribution
        self._sig_out_aver = np.average(sig_out, weights=na_clip, axis=1)

    def prepare_each(self, model, wngrid):
        """
        Interpolates the precomputed mie scattering opacity to the desired
        wavelength grid. Yields only once

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``BH`` and the weighted mie opacity.

        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        self.sigma_mie = np.interp(wngrid, self.wavenumberGrid,
                                   self._sig_out_aver)*self._mix_cloud_mix

        yield 'BH', self.sigma_mie

    def write(self, output):
        contrib = super().write(output)
        contrib.write_scalar('bh_cloud_particle_size', self._mie_radius)
        contrib.write_scalar('bh_cloud_topP', self._cloud_top_pressure)
        contrib.write_scalar('bh_cloud_bottomP', self._cloud_bottom_pressure)
        contrib.write_scalar('bh_cloud_mix', self._mix_cloud_mix)
        contrib.write_string('mie_path', self._mie_path)
        contrib.write_string('mie_type', self._mie_type)
        return contrib

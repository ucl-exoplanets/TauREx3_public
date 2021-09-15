from .contribution import Contribution
import numpy as np
from taurex.data.fittable import fitparam


class FlatMieContribution(Contribution):
    """
    Computes a flat absorption contribution
    across all wavelengths to the optical depth

    Parameters
    ----------

    flat_mix_ratio: float
        Opacity value

    flat_bottomP: float
        Bottom of absorbing region in Pa

    flat_topP: float
        Top of absorbing region in Pa

    """

    def __init__(self,
                 flat_mix_ratio=1e-10, flat_bottomP=-1,
                 flat_topP=-1):
        super().__init__('Mie')

        self._mie_mix = flat_mix_ratio
        self._mie_bottom_pressure = flat_bottomP
        self._mie_top_pressure = flat_topP

    @fitparam(param_name='flat_topP',
              param_latex='$P^{mie}_\mathrm{top}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[1e-20, 1])
    def mieTopPressure(self):
        """
        Pressure at top of absorbing region in Pa
        """
        return self._mie_top_pressure

    @mieTopPressure.setter
    def mieTopPressure(self, value):
        self._mie_top_pressure = value

    @fitparam(param_name='flat_bottomP',
              param_latex='$P^{mie}_\mathrm{bottom}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[1e-20, 1])
    def mieBottomPressure(self):
        """
        Pressure at bottom of absorbing region in Pa
        """
        return self._mie_bottom_pressure

    @mieBottomPressure.setter
    def mieBottomPressure(self, value):
        self._mie_bottom_pressure = value

    @fitparam(param_name='flat_mix_ratio',
              param_latex='$\chi_\mathrm{mie}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[1e-20, 1])
    def mieMixing(self):
        """
        Opacity of absorbing region in m2
        """
        return self._mie_mix

    @mieMixing.setter
    def mieMixing(self, value):
        self._mie_mix = value

    def prepare_each(self, model, wngrid):
        """
        Computes and flat absorbing opacity for
        the pressure regions given

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Flat`` and the weighted mie opacity.


        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_levels = np.log10(model.pressure.pressure_profile_levels[::-1])

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:

            bottom_pressure = pressure_levels.max()

        top_pressure = np.log10(self.mieTopPressure)
        if top_pressure < 0:
            top_pressure = pressure_levels.min()

        P_left = pressure_levels[:-1]
        P_right = pressure_levels[1:]

        P_range = sorted([top_pressure, bottom_pressure])

        save_start = np.searchsorted(P_right, P_range[0], side='right')
        save_stop = np.searchsorted(P_left[1:], P_range[1], side='right')
        P_min = P_left[save_start:save_stop+1]
        P_max = P_right[save_start:save_stop+1]
        weight = np.minimum(P_range[-1], P_max) - np.maximum(P_range[0], P_min)
        weight /= weight.max()
        sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))
        sigma_xsec[save_start:save_stop+1] = weight[:,  None]*self.mieMixing

        sigma_xsec = sigma_xsec[::-1]

        self.sigma_xsec = sigma_xsec

        yield 'Flat', sigma_xsec

    def write(self, output):
        contrib = super().write(output)
        contrib.write_scalar('flat_mix_ratio', self._mie_mix)
        contrib.write_scalar('flat_bottomP', self._mie_bottom_pressure)
        contrib.write_scalar('flat_topP', self._mie_top_pressure)
        return contrib

    @classmethod
    def input_keywords(self):
        return ['FlatMie', ]
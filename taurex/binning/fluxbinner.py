from .binner import Binner
from taurex.util.util import compute_bin_edges
import numpy as np
from taurex import OutputSize


class FluxBinner(Binner):
    """
    Bins to a wavenumber grid given by ``wngrid`` using a
    more accurate method that takes into account the amount
    of contribution from each native bin. This method also
    handles cases where bins are not continuous and/or
    overlapping.

    Parameters
    ----------

    wngrid: :obj:`array`
        Wavenumber grid

    wngrid_width: :obj:`array`, optional
        Must have same shape as ``wngrid``
        Full bin widths for each wavenumber grid point
        given in ``wngrid``. If not provided then
        this is automatically computed from ``wngrid``.

    """
    def __init__(self, wngrid, wngrid_width=None):
        super().__init__()

        sort_grid = wngrid.argsort()
        self._wngrid = wngrid[sort_grid]
        self._wngrid_width = wngrid_width

        if self._wngrid_width is None:
            self._wngrid_width = compute_bin_edges(self._wngrid)[-1]
        elif hasattr(self._wngrid_width, '__len__'):
            if len(self._wngrid_width) != len(self._wngrid):
                raise ValueError('Wavenumber width should be signel value or '
                                 'same shape as wavenumber grid')
            self._wngrid_width = wngrid_width[sort_grid]

        if not hasattr(self._wngrid_width, '__len__'):
            self._wngrid_width = np.ones_like(self._wngrid)*self._wngrid_width

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        """

        Bins down spectrum.

        Parameters
        ----------
        wngrid : :obj:`array`
            The wavenumber grid of the spectrum to be binned down.

        spectrum: :obj:`array`
            The spectra we wish to bin-down. Must be same shape as
            ``wngrid``.

        grid_width: :obj:`array`, optional
            Wavenumber grid full-widths for the spectrum to be binned down.
            Must be same shape as ``wngrid``.
            Optional.

        error: :obj:`array`, optional
            Associated errors or noise of the spectrum. Must be same shape
            as ``wngrid``.Optional parameter.

        Returns
        -------
        binned_wngrid : :obj:`array`
            New wavenumber grid

        spectrum: :obj:`array`
            Binned spectrum.

        grid_width: :obj:`array`
            New grid-widths

        error: :obj:`array` or None
            Binned error if given else ``None``

        """

        sorted_input = wngrid.argsort()
        spectrum = spectrum[..., sorted_input]
        if error is not None:
            error = error[..., sorted_input]

        bin_spectrum = np.zeros(spectrum[..., 0].shape + self._wngrid.shape)

        if error is not None:
            bin_error = np.zeros(spectrum[..., 0].shape + self._wngrid.shape)
        else:
            bin_error = None

        old_spect_wn = wngrid

        old_spect_flux = spectrum
        old_spect_err = error

        old_spect_width = grid_width

        if old_spect_width is None:
            old_spect_width = compute_bin_edges(old_spect_wn)[-1]

        old_spect_min = old_spect_wn - old_spect_width/2
        old_spect_max = old_spect_wn + old_spect_width/2

        new_spec_lhs = self._wngrid
        new_spec_rhs = self._wngrid_width

        new_spec_wn = self._wngrid

        new_spec_wn_min = new_spec_lhs - new_spec_rhs/2
        new_spec_wn_max = new_spec_lhs + new_spec_rhs/2

        save_start = 0
        save_stop = 0

        for idx, res in enumerate(zip(new_spec_wn, new_spec_wn_min,
                                      new_spec_wn_max)):

            wn, wn_min, wn_max = res
            sum_spectrum = 0
            sum_noise = 0
            sum_weight = 0

            save_start = np.searchsorted(old_spect_max, wn_min, side='right')
            save_stop = np.searchsorted(old_spect_min[1:], wn_max,
                                        side='right')

            save_stop = min(save_stop, old_spect_min.shape[0]-1)
            save_start = min(save_start, old_spect_min.shape[0]-1)

            if not wn_min <= old_spect_max[save_start] or not \
                    old_spect_min[save_stop] <= wn_max:
                continue

            spect_min = old_spect_min[save_start:save_stop+1]
            spect_max = old_spect_max[save_start:save_stop+1]

            weight = (np.minimum(wn_max, spect_max) -
                      np.maximum(spect_min, wn_min))/(wn_max-wn_min)

            sum_weight = np.sum(weight)

            sum_spectrum = np.sum(weight *
                                  old_spect_flux[..., save_start:save_stop+1],
                                  axis=-1)

            if error is not None:
                sum_noise = np.sum(weight * weight *
                                   old_spect_err[..., save_start:save_stop+1]**2,
                                   axis=0)

                sum_noise = np.sqrt(sum_noise / sum_weight/sum_weight)

            bin_spectrum[..., idx] = sum_spectrum

            if error is not None:
                bin_error[idx] = sum_noise

        return self._wngrid, bin_spectrum, bin_error, self._wngrid_width

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):

        output = super().generate_spectrum_output(model_output,
                                                  output_size=output_size)
        output['binned_wngrid'] = self._wngrid
        output['binned_wlgrid'] = 10000/self._wngrid
        output['binned_wnwidth'] = self._wngrid_width
        output['binned_wlwidth'] = 1.0/self._wngrid_width
        return output

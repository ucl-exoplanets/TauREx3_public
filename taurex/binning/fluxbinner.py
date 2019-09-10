from .binner import Binner
from taurex.util.util import compute_bin_edges
import numpy as np


class FluxBinner(Binner):

    def __init__(self, wngrid, wngrid_width=None):
        super().__init__()
        self._wngrid = wngrid
        self._wngrid_width = wngrid_width

        if self._wngrid_width is None:
            self._wngrid_width = compute_bin_edges(self._wngrid)[-1]

        if not hasattr(self._wngrid_width,'__len__'):
            self._wngrid_width = np.ones_like(self._wngrid)*self._wngrid_width

        if self._wngrid_width.shape != self._wngrid.shape:
            raise ValueError('Wavenumber width should be signel value or same shape as wavenumber grid')


        self._wngrid = self._wngrid[self._wngrid.argsort()]
        self._wngrid_width = self._wngrid_width[self._wngrid.argsort()]

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        sorted_input = wngrid.argsort()
        spectrum = spectrum[...,sorted_input]
        if error is not None:
            error = error[...,sorted_input]

        bin_spectrum = np.zeros(spectrum[...,0].shape + self._wngrid.shape)

        if error is not None:
            bin_error = np.zeros(spectrum[...,0].shape + self._wngrid.shape)
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

            try:
                while old_spect_max[save_start] <= wn_min:
                    save_start += 1
            except IndexError:
                save_start = old_spect_max.shape[-1]-1
            try:

                while old_spect_min[save_stop+1] <= wn_max:
                    save_stop += 1

            except IndexError:
                save_stop = old_spect_max.shape[-1]-1

            if not wn_min <= old_spect_max[save_start] or not old_spect_min[save_stop] <= wn_max:
                continue

            spect_min = old_spect_min[save_start:save_stop+1]
            spect_max = old_spect_max[save_start:save_stop+1]

            weight = np.zeros_like(old_spect_min[save_start:save_stop+1])

            weight[...] = 0.0

            hw_min_lt_min = spect_min <= wn_min
            hw_max_gt_min = spect_max >= wn_min

            hw_max_gt_max = spect_max >= wn_max

            hw_min_lt_max = spect_min <= wn_max

            hw_not_min = ~hw_min_lt_min

            branch_filt = hw_min_lt_min & hw_max_gt_min & hw_max_gt_max

            weight[branch_filt] = wn_max - wn_min

            branch_filt = hw_min_lt_min & hw_max_gt_min & ~hw_max_gt_max

            weight[branch_filt] = spect_max[branch_filt] - wn_min

            branch_filt = hw_not_min & hw_min_lt_max & hw_max_gt_max

            weight[branch_filt] = wn_max - spect_min[branch_filt]

            branch_filt = hw_not_min & hw_min_lt_max & ~hw_max_gt_max

            weight[branch_filt] = spect_max[branch_filt] - spect_min[branch_filt]

            weight /= (wn_max-wn_min)

            #weight /= weight.sum()
            sum_weight = np.sum(weight)

            sum_spectrum = np.sum(weight*old_spect_flux[...,save_start:save_stop+1],
                                axis=-1)


            if error is not None:
                sum_noise = np.sum(weight * weight *
                                old_spect_err[..., save_start:save_stop+1]**2,
                                axis=0)
                

                sum_noise = np.sqrt(sum_noise / sum_weight/sum_weight)

            bin_spectrum[...,idx] = sum_spectrum

            if error is not None:
                bin_error[idx] = sum_noise

        return self._wngrid, bin_spectrum, bin_error, self._wngrid_width
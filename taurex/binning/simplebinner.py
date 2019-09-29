from .binner import Binner
from taurex.util import bindown
from taurex.util.util import compute_bin_edges,wnwidth_to_wlwidth
from taurex import OutputSize


class SimpleBinner(Binner):

    def __init__(self, wngrid, wngrid_width=None):

        self._wngrid = wngrid
        self._wn_width = wngrid_width or compute_bin_edges(self._wngrid)[-1]

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):

        return self._wngrid, bindown(wngrid, spectrum, self._wngrid), None, self._wn_width

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):

        output = super().generate_spectrum_output(model_output,
                                                  output_size=output_size)
        output['binned_wngrid'] = self._wngrid
        output['binned_wlgrid'] = 10000/self._wngrid
        output['binned_wnwidth'] = self._wn_width
        output['binned_wlwidth'] = wnwidth_to_wlwidth(self._wn_width,self._wn_width)
        return output

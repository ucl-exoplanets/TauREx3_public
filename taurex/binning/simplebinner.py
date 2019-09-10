from .binner import Binner
from taurex.util import bindown
from taurex.util.util import compute_bin_edges


class SimpleBinner(Binner):

    def __init__(self, wavenumber_grid, wngrid_width=None):

        self._wngrid = wavenumber_grid
        self._wn_wdith = wngrid_width or compute_bin_edges(self._wngrid)


    def bindown(self, wngrid, spectrum, grid_width=None, error=None):

            return self._wngrid, bindown(wngrid, spectrum, self._wngrid), None, self._wn_wdith


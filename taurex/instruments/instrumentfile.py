from .instrument import Instrument
import numpy as np
from taurex.binning import FluxBinner
import math
from taurex.util.util import wnwidth_to_wlwidth


class InstrumentFile(Instrument):
    """
    Loads a 2-3 column file with wlgrid
    and noise and maybe wlgrid


    Parameters
    ----------

    filename: str
        Filename of file containing the binning and error


    """

    def __init__(self, filename=None):
        super().__init__()

        self._spectrum = np.loadtxt(filename)

        self._wlgrid = self._spectrum[:, 0]
        
        sortedwl = self._wlgrid.argsort()[::-1]

        self._wlgrid = self._wlgrid[sortedwl]

        self._wngrid = 10000/self._wlgrid

        self._noise = self._spectrum[sortedwl, 1]

        try:
            self._wlwidths = self._spectrum[sortedwl, 2]
        except IndexError:
            from taurex.util.util import compute_bin_edges

            self._wlwidths - compute_bin_edges(self._wlgrid)[-1]

        self.create_wn_widths()

        self._binner = FluxBinner(self._wngrid, wngrid_width=self._wnwidths)

    def create_wn_widths(self):

        self._wnwidths = wnwidth_to_wlwidth(self._wlgrid, self._wlwidths)

    def model_noise(self, model, model_res=None, num_observations=1):

        if model_res is None:
            model_res = model.model()

        wngrid, spectrum, error, grid_width = self._binner.bin_model(model_res)

        return wngrid, spectrum, self._noise / math.sqrt(num_observations), grid_width

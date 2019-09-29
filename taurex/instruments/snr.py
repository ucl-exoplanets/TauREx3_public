from .instrument import Instrument
import numpy as np
from taurex.binning import FluxBinner
import math


class SNRInstrument(Instrument):
    """



    Parameters
    ----------

    filename: str
        Filename of file containing the binning and error


    """

    def __init__(self, SNR=10, binner=None):
        super().__init__()

        self._binner = binner
        self._SNR = SNR

    def model_noise(self, model, model_res=None, num_observations=1):

        if model_res is None:
            model_res = model.model()




        wngrid, spectrum, error, grid_width = self._binner.bin_model(model_res)

        signal = spectrum.max() - spectrum.min()

        noise = np.ones(spectrum.shape)*signal/self._SNR

        return wngrid, spectrum, noise / math.sqrt(num_observations), grid_width

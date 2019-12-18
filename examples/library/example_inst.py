from taurex.instruments import Instrument
from taurex.binning import FluxBinner
import numpy as np
from taurex.util.util import wnwidth_to_wlwidth


class ExampleInstrument(Instrument):
    """
    An example implementation of an instrument
    An instument function thats uses the WFC3
    spectral grid and applies a gaussian noise
    with a scale ``noise_scale`` for each spectrum
    point.

    """
    def __init__(self, noise_scale=1):
        super().__init__()

        self._scale = noise_scale

        # Wavelength and widths for WFC3
        wfc3_grid = np.array([
            1.126, 1.156, 1.184, 1.212, 1.238, 1.265, 1.292, 1.318, 1.345, 
            1.372, 1.399, 1.428, 1.457, 1.487, 1.518, 1.551, 1.586, 1.623,
        ])

        wfc3_wlwidths = np.array([
            3.079e-2, 2.930e-2, 2.790e-2, 2.689e-2, 2.649e-2, 2.689e-2,
            2.670e-2, 2.629e-2, 2.649e-2, 2.739e-2, 2.800e-2, 2.849e-2,
            2.940e-2, 3.079e-2, 3.180e-2, 3.370e-2, 3.600e-2, 3.899e-2,
        ])

        # convert to wavenumber widths
        wfc3_wnwidths = wnwidth_to_wlwidth(wfc3_grid, wfc3_wlwidths)

        self._wfc3_size = wfc3_grid.shape[0]

        # Create our grid resampler
        self._binner = FluxBinner(wngrid=10000/wfc3_grid,
                                  wngrid_width=wfc3_wnwidths)

    def model_noise(self, model, model_res=None, num_observations=1):

        if model_res is None:
            model_res = model.model()

        # Generate our noise
        noise = np.random.normal(scale=self._scale, size=self._wfc3_size)

        # Bin down our model
        wngrid, depth, _, widths = self._binner.bin_model(model_res)

        return wngrid, depth, noise, widths

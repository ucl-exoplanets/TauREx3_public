from taurex.log import Logger
from taurex import OutputSize
from taurex.util.util import compute_bin_edges


class Binner(Logger):

    def __init__(self):
        Logger.__init__(self, self.__class__.__name__)
    
    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        raise NotImplementedError

    def bin_model(self, model_output):
        return self.bindown(model_output[0], model_output[1])

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):
        output = {}

        wngrid, flux, tau, extra = model_output

        output['native_wngrid'] = wngrid
        output['native_wlgrid'] = 10000/wngrid
        output['native_spectrum'] = flux
        output['binned_spectrum'] = self.bindown(wngrid, flux)[1]
        output['native_wnwidth'] = compute_bin_edges(wngrid)[-1]
        output['native_wlwidth'] = compute_bin_edges(10000/wngrid)[-1]
        if output_size > OutputSize.lighter:
            output['binned_tau'] = self.bindown(wngrid, tau)[1]
            if output_size > OutputSize.light:
                output['native_tau'] = tau

        return output

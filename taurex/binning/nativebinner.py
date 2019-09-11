from taurex.log import Logger
from taurex import OutputSize


class NativeBinner(Logger):

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        return wngrid, spectrum, error, grid_width

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):
        output = {}

        wngrid, flux, tau, extra = model_output

        output['native_wngrid'] = wngrid
        output['native_wlgrid'] = 10000/wngrid
        output['native_spectrum'] = flux

        if output_size > OutputSize.light:
            output['native_tau'] = tau

        return output

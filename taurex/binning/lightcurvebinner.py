from .binner import Binner
from taurex import OutputSize


class LightcurveBinner(Binner):
    """
    A special class of binning used to generate the correct spectrum output
    """

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        return wngrid, spectrum, error, grid_width

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):
        output = {}

        wngrid, lightcurve, tau, extra = model_output
        native_grid, native, binned, extra = extra

        
        output['native_wngrid'] = native_grid
        output['native_wlgrid'] = 10000/native_grid
        output['binned_wngrid'] = wngrid
        output['binned_wlgrid'] = 10000/wngrid
        output['lightcurve'] = lightcurve
        output['native_spectrum'] = native
        output['binned_spectrum'] = binned
        if output_size > OutputSize.lighter:
            output['binned_tau'] = self.bindown(wngrid, tau)[1]
            if output_size > OutputSize.light:
                output['native_tau'] = tau

        return output

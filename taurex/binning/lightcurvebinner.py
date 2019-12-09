from .binner import Binner
from taurex import OutputSize


class LightcurveBinner(Binner):
    """
    A special class of binning used to generate the correct spectrum output.
    This is essentially the same as 
    :class:`~taurex.binning.nativebinner.NativeBinner`
    but for lightcurve forward models.


    """

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        """Does nothing, only returns function arguments"""
        return wngrid, spectrum, error, grid_width

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):
        """
        Accepts only a lightcurve forward model. Stores the lightcurve
        as well as the spectrum.

        Parameters
        ----------
        model_output: obj:`tuple`
            Result from running a TauREx3 lightcurve forward model

        output_size: :class:`~taurex.taurexdefs.OutputSize`
            Size of the output.


        Returns
        -------
        :obj:`dict`:
            Dictionary of spectra containing both lightcurves
            and spectra.


        """

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

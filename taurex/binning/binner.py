"""Module for the base binning class"""

from taurex.log import Logger
from taurex import OutputSize
from taurex.util.util import compute_bin_edges


class Binner(Logger):
    """
    *Abstract class*

    The binner class deals with binning down spectra to different resolutions.
    It also provides a method to generate spectrum output format from a forward
    model result in the form of a dictionary.
    Using this class does not need to be restricted to TauREx3 results and can
    be used to bin down any arbitrary spectra.
    """
    def __init__(self):
        Logger.__init__(self, self.__class__.__name__)

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        """
        **Requires implementation**

        This should handle the binning of a spectrum passed into the function.
        Parameters given are guidelines on expectation of usage.

        Parameters
        ----------
        wngrid : :obj:`array`
            The wavenumber grid of the spectrum to be binned down.
            Generally the 'native' wavenumber grid

        spectrum: :obj:`array`
            The spectra we wish to bin-down. Must be same shape as
            ``wngrid``.

        grid_width: :obj:`array`, optional
            Wavenumber grid full-widths for the spectrum to be binned down.
            Must be same shape as ``wngrid``.
            Optional, generally if you require this but the user does not pass
            it then you must compute it yourself using ``wngrid``. This can
            be done easily using the function
            func:`~taurex.util.util.compute_bin_edges`.

        error: :obj:`array`, optional
            Associated errors or noise of the spectrum. Must be same shape
            as ``wngrid``.Optional parameter, when implementing you must
            deal with the cases where either the error is passed or not passed.

        Returns
        -------
        binned_wngrid : :obj:`array`
            New wavenumber grid

        spectrum: :obj:`array`
            Binned spectrum.

        grid_width: :obj:`array`
            New grid-widths

        error: :obj:`array` or None
            If passed, should be the binned error otherwise None

        """
        raise NotImplementedError

    def bin_model(self, model_output):
        """
        Bins down a TauREx3 forward model.
        This automatically splits the output and passes it to
        the :func:`bindown` function.
        Its general usage is of the form:

        >>> fm = TransmissionModel()
        >>> fm.build()
        >>> result = fm.model()
        >>> binner.bin_model(result)

        Or in a single line:

        >>> binner.bin_model(fm.model())


        Parameters
        ----------
        model_output: obj:`tuple`
            Result from running a TauREx3 forward model
        
        Returns
        -------
        See :func:`bindown`


        """
        return self.bindown(model_output[0], model_output[1])

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):
        """
        Given a forward model output, generate a dictionary
        that can be used to store to file. This can include
        storing the native and binned spectrum.
        Not necessary for the function of the class but useful for
        full intergation into TauREx3, especially when storing results
        from a retrieval. 
        Can be overwritten to store more information.

        Parameters
        ----------
        model_output: obj:`tuple`
            Result from running a TauREx3 forward model

        output_size: :class:`~taurex.taurexdefs.OutputSize`
            Size of the output.


        Returns
        -------
        :obj:`dict`:
            Dictionary of spectra


        """
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

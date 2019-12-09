from .binner import Binner
from taurex.util import bindown
from taurex.util.util import compute_bin_edges, wnwidth_to_wlwidth
from taurex import OutputSize


class SimpleBinner(Binner):
    """
    Bins to a wavenumber grid given by ``wngrid``.
    The method places flux into the correct bins
    using histogramming methods. This is fast but can
    suffer as it assumes that there are no gaps in the
    wavenumber grid. This can cause weird results and
    may cause the flux to be higher in the boundary
    of points between two distinct regions
    (such as WFC3 + Spitzer)

    Parameters
    ----------

    wngrid: :obj:`array`
        Wavenumber grid

    wngrid_width: :obj:`array`, optional
        Must have same shape as ``wngrid``
        Full bin widths for each wavenumber grid point
        given in ``wngrid``. If not provided then
        this is automatically computed from ``wngrid``.

    """

    def __init__(self, wngrid, wngrid_width=None):

        self._wngrid = wngrid
        self._wn_width = wngrid_width
        if self._wn_width is None:
            self._wn_width = compute_bin_edges(self._wngrid)[-1]

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        """

        Bins down spectrum.

        Parameters
        ----------
        wngrid : :obj:`array`
            The wavenumber grid of the spectrum to be binned down.

        spectrum: :obj:`array`
            The spectra we wish to bin-down. Must be same shape as
            ``wngrid``.

        grid_width: :obj:`array`, optional
            Wavenumber grid full-widths for the spectrum to be binned down.
            Must be same shape as ``wngrid``.
            Optional.

        error: :obj:`array`, optional
            Associated errors or noise of the spectrum. Must be same shape
            as ``wngrid``.Optional parameter.

        Returns
        -------
        binned_wngrid : :obj:`array`
            New wavenumber grid

        spectrum: :obj:`array`
            Binned spectrum.

        grid_width: :obj:`array`
            New grid-widths

        error: :obj:`array` or None
            Binned error if given else ``None``

        """
        return self._wngrid, bindown(wngrid, spectrum, self._wngrid), \
            None, self._wn_width

    def generate_spectrum_output(self, model_output,
                                 output_size=OutputSize.heavy):

        output = super().generate_spectrum_output(model_output,
                                                  output_size=output_size)
        output['binned_wngrid'] = self._wngrid
        output['binned_wlgrid'] = 10000/self._wngrid
        output['binned_wnwidth'] = self._wn_width
        output['binned_wlwidth'] = wnwidth_to_wlwidth(self._wngrid,
                                                      self._wn_width)
        return output

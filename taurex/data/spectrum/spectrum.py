
"""
Contains the basic definition of an observed spectrum for TauRex 3
"""

from taurex.log import Logger
from taurex.output.writeable import Writeable


class BaseSpectrum(Logger, Writeable):
    """

    *Abstract class*

    A base class where spectrums are loaded (or later created). This
    is used to either plot against the forward model or passed into the 
    optimizer to be used to fit the forward model

    Parameters
    ----------

    name : str
        Name to be used in logging

    """

    def __init__(self, name):
        super().__init__(name)

    def create_binner(self):
        """
        Creates the appropriate binning object
        """
        from taurex.binning import FluxBinner

        return FluxBinner(wngrid=self.wavenumberGrid,
                          wngrid_width=self.binWidths)

    @property
    def spectrum(self):
        """
        **Requires Implementation**


        Should return the observed spectrum.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def rawData(self):
        """
        **Requires Implementation**


        Should return the raw data set.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def wavelengthGrid(self):
        """
        **Requires Implementation**


        Should return the wavelength grid of the spectrum in microns.
        This does not need to necessarily match the shape of :func:`spectrum`

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def wavenumberGrid(self):
        """
        Wavenumber grid in cm-1

        Returns
        -------
        wngrid : :obj:`array`

        """
        return 10000/self.wavelengthGrid

    @property
    def binEdges(self):
        """
        **Requires Implementation**


        Should return the bin edges of the wavenumber grid

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def binWidths(self):
        """
        **Requires Implementation**


        Should return the widths of each bin in the wavenumber grid

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def errorBar(self):
        """
        **Requires Implementation**


        Should return the error. *Must* be the same shape as
        :func:`spectrum`

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def write(self, output):
        output.write_array('wlgrid', self.wavelengthGrid)
        output.write_array('spectrum', self.spectrum)
        output.write_array('binedges', self.binEdges)
        output.write_array('binwidths', self.binWidths)
        output.write_array('errorbars', self.errorBar)

        return output

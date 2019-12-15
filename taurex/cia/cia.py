"""
Contains the abstract class used by all collisionally induced absorption
objects.
"""

from taurex.log import Logger
import numpy as np


class CIA(Logger):
    """

    *Abstract class*

    This is the base class for collisionally induced absorption opacities.
    To function in Taurex3, it requires concrete implementations of:
        - :func:`wavenumberGrid`
        - :func:`compute_cia`
        - :func:`temperatureGrid`



    Parameters
    ----------
    name : str
        Name to use for logging

    pair_name : str
        pair of molecules this class represents. e.g. 'H2-H2' or 'H2-He'

    """

    def __init__(self, name, pair_name):
        super().__init__(name)

        self._pair_name = pair_name

    @property
    def pairName(self):
        """
        The assigned pair of molecules of this CIA

        Returns
        -------
        str
            The pair of molecules of this object in the form: ``Molecule1``-``Molecule2``

        """
        return self._pair_name


    @property
    def pairOne(self):
        """
        The name of the first molecule in the pair

        Returns
        -------
        str
            First molecule in the pair

        """

        return self._pair_name.split('-')[0]

    @property
    def pairTwo(self):
        """
        The name of the second molecule in the pair

        Returns
        -------
        str
            Second molecule in the pair

        """

        return self._pair_name.split('-')[-1]

    def compute_cia(self, temperature):
        """
        Computes the collisionaly induced cross-section for a given temeprature

        Unimplemented, this must be implemented in any derived class to be
        considered compatible in Taurex3

        The rules are:
            1. It must accept temperature in Kelvin (K)
            2. If the temperature falls outside of :func:`temperatureGrid` it must be set to zero
            3. The returned array must be of equal size to :func:`wavenumberGrid`


        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        Returns
        -------
        :obj:`array`
            CIA cross section at desired temeprature on its native grid

        Raises
        ------
        NotImplementedError
            Only if derived class does not implement this

        """
        raise NotImplementedError

    @property
    def wavenumberGrid(self):
        """
        The native wavenumber grid (cm-1) of the CIA. Must be implemented in
        derived classes

        Returns
        -------
        :obj:`array`
            Native wavenumber grid

        Raises
        ------
        NotImplementedError
            Only if derived class does not implement this

        """
        raise NotImplementedError

    @property
    def temperatureGrid(self):

        """
        The native temperature grid of the CIA cross-sections.

        Returns
        -------
        :obj:`array`
            Native temeprature grid in Kelvin

        Raises
        ------
        NotImplementedError
            Only if derived class does not implement this

        """

        raise NotImplementedError

    def cia(self, temperature, wngrid=None):
        """
        For a given temperature, computes the appropriate cross section.
        If wavenumber grid ( :obj:`wngrid` ) is provided then the
        cross-section is interpolated
        to it.

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        wngrid : :obj:`array` , optional
            Wavenumber grid to interpolate to


        Returns
        -------
        :obj:`array`
            CIA cross section at desired temeprature on either its native grid
            or interpolated on :obj:`wngrid` if supplied

        """

        orig = self.compute_cia(temperature)
        if wngrid is None:
            return orig
        else:
            return np.interp(wngrid, self.wavenumberGrid, orig)

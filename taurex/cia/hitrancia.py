"""Module contains classes that handle loading of HITRAN cia files"""
from .cia import CIA
import numpy as np
from taurex.util.math import interp_lin_only


class EndOfHitranCIAException(Exception):
    """
    An exception that occurs when the end of a HITRAN file
    is reached
    """
    pass


def hashwn(start_wn, end_wn):
    """
    Simple wavenumber hash function
    """
    return str(start_wn)+str(end_wn)


class HitranCiaGrid(object):
    """
    Class that handles a particular HITRAN cia wavenumber grid
    Since temperatures for CIA sometimes have different wavenumber grids this
    class helps to simplify managing them by only dealing with one at a time.
    These will help us unify into a single grid eventually

    Parameters
    ----------
    wn_min : float
        The minimum wavenumber for this grid

    wn_max : float
        The maximum wavenumber for this grid

    """

    def __init__(self, wn_min, wn_max):
        self.wn = None
        self.Tsigma = []

    def add_temperature(self, T, sigma):
        """
        Adds a temeprature and crossection to this wavenumber grid

        Parameters
        ----------
        T : float
            Temeprature in Kelvin

        sigma : :obj:`array`
            cross-sections for this grid

        """

        self.Tsigma.append((T, sigma))

    @property
    def temperature(self):
        """

        Gets the current temeprature grid for this wavenumber grid

        Returns
        -------
        :obj:`array`
            Temeprature grid in Kelvin

        """
        return [t for t, s in self.Tsigma]

    @property
    def sigma(self):
        """
        Gets the currently loaded crossections for this wavenumber grid

        Returns
        -------
        :obj:`array`
            Cross-section grid

        """
        return [s for t, s in self.Tsigma]

    def find_closest_temperature_index(self, temperature):
        """
        Finds the nearest indices for a particular temperature

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        Returns
        -------
        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        """
        t_min = np.array(self.temperature).searchsorted(temperature,
                                                        side='right')-1
        t_max = t_min+1
        return t_min, t_max

    def interp_linear_grid(self, T, t_idx_min, t_idx_max):
        """
        For a given temperature and indicies. Interpolate the cross-sections
        linearly from temperature grid to temperature ``T``

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin
        
        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        Returns
        -------
        out : :obj:`array`
            Interpolated cross-section

        """

        temp_grid = np.array(self.temperature)
        Tmax = temp_grid[t_idx_max]
        Tmin = temp_grid[t_idx_min]
        fx0 = self.sigma[t_idx_min]
        fx1 = self.sigma[t_idx_max]

        return interp_lin_only(fx0, fx1, T, Tmin, Tmax)

    def sortTempSigma(self):
        """
        Sorts the temperature-sigma list
        """
        import operator
        self.Tsigma.sort(key=operator.itemgetter(0))

    def fill_temperature(self, temperatures):
        """

        Here the 'master' temperature grid is passed into here and
        any gaps in our grid is filled with zero cross-sections to produce
        our final temperature-crosssection grid that matches with every other
        wavenumber grid. Temperatures that don't exist in the current grid but are withing
        the minimum and maximum for us are produced by linear interpolation


        Parameters
        ----------
        temperatures : array_like
            Master temperature grid

        """
        for t in temperatures:
            if t in self.temperature:
                continue
            self.debug('Tempurature %s, %s', t)
            if t < min(self.temperature) or t > max(self.temperature):
                self.add_temperature(t, np.zeros_like(self.wn))
            else:
                indicies = self.find_closest_temperature_index(t)
                self.add_temperature(t, self.interp_linear_grid(t, *indicies))
            self.sortTempSigma()


class HitranCIA(CIA):
    """

    A class that directly deals with HITRAN `cia <https://hitran.org/cia/>`_
    files and turns them into generic CIA objects that nicely produces
    cross sections for us.
    This will handle CIAs that have wavenumber grids split
    across temperatures by unifying
    them into single grids.

    To use it simply do:

    >>> h2h2 = HitranCIA('path/to/H2-He.cia')

    And now you can painlessly compute cross-sections like this:

    >>> h2h2.cia(400)

    Or if you have a wavenumber grid, we can also interpolate it:

    >>> h2h2.cia(400,mywngrid)

    And all it cost was buying me a beer!


    Parameters
    ----------
    filename : str
        Path to HITRAN cia file


    """

    def __init__(self, filename):
        super().__init__(self.__class__.__name__, 'None')

        self._filename = filename
        self._molecule_name = None
        self._wavenumber_grid = None
        self._temperature_grid = None
        self._xsec_grid = None
        self._wn_dict = {}
        self.load_hitran_file(filename)

    def load_hitran_file(self, filename):
        """
        Handles loading of the HITRAN file by reading and figuring
        out the wavenumber and temperature grids and matching them up

        Parameters
        ----------
        filename : str
            Path to HITRAN cia file

        """

        temp_list = []

        with open(filename, 'r') as f:
            # Read number of points
            while True:
                try:
                    start_wn, end_wn, total_points, T, \
                                max_cia = self.read_header(f)

                except EndOfHitranCIAException:
                    break
                if T not in temp_list:
                    temp_list.append(T)

                wn_hash = hashwn(start_wn, end_wn)

                wn_obj = None
                if wn_hash not in self._wn_dict:
                    self._wn_dict[wn_hash] = HitranCiaGrid(start_wn, end_wn)

                wn_obj = self._wn_dict[wn_hash]

                # Clear the temporary list
                sigma_temp = []
                wn_temp = []
                for x in range(total_points):
                    line = f.readline()
                    self.debug('Line %s', line)
                    splits = line.split()

                    _wn = splits[0]
                    _sigma = splits[1]
                    wn_temp.append(float(_wn))
                    _sig = float(_sigma)*1e-10
                    if _sig < 0:
                        _sig = 0
                    sigma_temp.append(_sig)

                # Ok we're done lets add the sigma
                wn_obj.add_temperature(T, np.array(sigma_temp))
                # set the wavenumber grid
                wn_obj.wn = np.array(wn_temp)

        temp_list.sort()
        self._temperature_grid = np.array(temp_list)
        self.fill_gaps(temp_list)
        self.compute_final_grid()

    def fill_gaps(self, temperature):
        """

        Fills gaps in temperature grid for all wavenumber grid objects
        we've created


        Parameters
        ----------
        temperature : array_like
            Master temperature grid


        """
        for wn_obj in self._wn_dict.values():
            wn_obj.sortTempSigma()
            wn_obj.fill_temperature(temperature)

    def compute_final_grid(self):
        """

        Collects all :class:`HitranCiaGrid` objects we've created
        and unifies them into a single temperature, cross-section and
        wavenumber grid for us to FINALLY interpolate and produce
        collisionaly induced cross-sections

        """

        _wngrid = []
        for w in self._wn_dict.values():
            _wngrid.append(w.wn)
        self._wavenumber_grid = np.concatenate(_wngrid)
        sorted_idx = np.argsort(self._wavenumber_grid)
        self._wavenumber_grid = self._wavenumber_grid[sorted_idx]
        _sigma_array = []

        for idx, t in enumerate(self._temperature_grid):
            _temp_sigma = []
            for w in self._wn_dict.values():
                _temp_sigma.append(w.Tsigma[idx][1])
            _sigma_array.append(np.concatenate(_temp_sigma)[sorted_idx])

        self._xsec_grid = np.array(_sigma_array)

    def find_closest_temperature_index(self, temperature):
        """
        Finds the nearest indices for a particular temperature

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        Returns
        -------
        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        """
        t_min = self.temperatureGrid.searchsorted(temperature, side='right')-1
        t_max = t_min+1
        return t_min, t_max

    def interp_linear_grid(self, T, t_idx_min, t_idx_max):
        """
        For a given temperature and indicies. Interpolate the cross-sections
        linearly from temperature grid to temperature ``T``

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        Returns
        -------
        out : :obj:`array`
            Interpolated cross-section

        """
        if T > self._temperature_grid.max():
            return self._xsec_grid[-1]
        elif T < self._temperature_grid.min():
            return self._xsec_grid[0]

        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        fx0 = self._xsec_grid[t_idx_min]
        fx1 = self._xsec_grid[t_idx_max]

        return interp_lin_only(fx0, fx1, T, Tmin, Tmax)


    def read_header(self, f):
        """
        Reads single header in the file

        Parameters
        ----------
        f : file object

        Returns
        -------
        start_wn : float
            Start wavenumber for temperature

        end_wn : float
            End wavenumber for temperature

        total_points : int
            total number of points in temperature

        T : float
            Temperature in Kelvin

        max_cia : float
            Maximum CIA value in temperature


        """

        line = f.readline()
        if line is None or line == '':
            raise EndOfHitranCIAException
        split = line.split()
        self._pair_name = split[0]
        start_wn = float(split[1])
        end_wn = float(split[2])
        total_points = int(split[3])
        T = float(split[4])
        max_cia = float(split[5])

        return start_wn, end_wn, total_points, T, max_cia

    @property
    def wavenumberGrid(self):
        """
        Unified wavenumber grid

        Returns
        -------
        :obj:`array`
            Native wavenumber grid

        """

        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        """
        Unified temperature grid

        Returns
        -------
        :obj:`array`
            Native temperature grid in Kelvin

        """
        return self._temperature_grid

    def compute_cia(self, temperature):
        """
        Computes the collisionally induced absorption cross-section
        using our final native temperature and cross-section grids

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        out : :obj:`array`
            Temperature interpolated cross-section

        """

        indicies = self.find_closest_temperature_index(temperature)

        return self.interp_linear_grid(temperature, *indicies)


from .star import BlackbodyStar
import numpy as np
import os
from taurex.constants import MSOL
import math


class PhoenixStar(BlackbodyStar):
    """
    A star that uses the `PHOENIX <https://www.aanda.org/articles/aa/abs/2013/05/aa19058-12/aa19058-12.html>`_
    synthetic stellar atmosphere spectrums.

    These spectrums are read from ``.gits.gz`` files in a directory given by
    ``phoenix_path``
    Each file must contain the spectrum for one temperature

    Parameters
    ----------

    phoenix_path: str, **required**
        Path to folder containing phoenix ``fits.gz`` files

    temperature: float, optional
        Stellar temperature in Kelvin

    radius: float, optional
        Stellar radius in Solar radius

    metallicity: float, optional
        Metallicity in solar values

    mass: float, optional
        Stellar mass in solar mass

    distance: float, optional
        Distance from Earth in pc

    magnitudeK: float, optional
        Maginitude in K band


    Raises
    ------
    Exception
        Raised when no phoenix path is defined


    """

    def __init__(self, temperature=5000, radius=1.0, metallicity=1.0, mass=1.0,
                 distance=1, magnitudeK=10.0, phoenix_path=None):
        super().__init__(temperature=temperature, radius=radius,
                         distance=distance,
                         magnitudeK=magnitudeK, mass=mass,
                         metallicity=metallicity)
        if phoenix_path is None:
            self.error('No file path to phoenix files defined')
            raise Exception('No file path to phoenix files defined')

        self.info('Star is PHOENIX type')
        self._phoenix_path = phoenix_path

        self.get_avail_phoenix()
        self.use_blackbody = False
        self.recompute_spectra()
        # self.preload_phoenix_spectra()

    def compute_logg(self):
        """
        Computes log(surface_G)

        """
        import astropy.units as u
        from astropy.constants import G
        mass = self._mass * u.kg
        radius = self._radius * u.m

        small_g = (G * mass) / (radius**2)

        small_g = small_g.to(u.cm/u.s**2)

        return math.log10(small_g.value)

    def recompute_spectra(self):

        if self.temperature > self._T_list.max() or \
                self.temperature < self._T_list.min():

            self.use_blackbody = True
        else:
            self.use_blackbody = False
            self._logg = self.compute_logg()
            f = self.find_nearest_file()
            self.read_spectra(f)

    def read_spectra(self, p_file):
        from astropy.io import fits
        import astropy.units as u

        with fits.open(p_file) as hdu:
            strUnit = hdu[1].header['TUNIT1']
            wl = hdu[1].data.field('Wavelength') * u.Unit(strUnit)

            strUnit = hdu[1].header['TUNIT2']
            sed = hdu[1].data.field('Flux') * u.Unit(strUnit)

            self.wngrid = 10000/(wl.value)
            argidx = np.argsort(self.wngrid)
            self._base_sed = sed.to(u.W/u.m**2/u.micron)
            self.wngrid = self.wngrid[argidx]
            self._base_sed = self._base_sed[argidx]

    @property
    def temperature(self):
        """
        Effective Temperature in Kelvin

        Returns
        -------
        T: float

        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.recompute_spectra()

    @property
    def mass(self):
        """
        Mass of star in solar mass

        Returns
        -------
        M: float

        """
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = value * MSOL
        self.recompute_spectra()

    def find_nearest_file(self):

        idx = self._index_finder(
            [self._temperature, self._logg, self._metallicity])
        return self._files[int(idx)]

    def get_avail_phoenix(self):
        from scipy.interpolate import NearestNDInterpolator
        import glob
        files = glob.glob(os.path.join(self._phoenix_path, '*.spec.fits.gz'))
        self._files = files
        self._T_list = np.array(
            [np.float(os.path.basename(k)[3:8]) for k in files])*100
        self._Logg_list = np.array(
            [np.float(os.path.basename(k)[9:12]) for k in files])
        self._Z_list = np.array(
            [np.float(os.path.basename(k)[13:16]) for k in files])
        self._index_finder = NearestNDInterpolator(
            (self._T_list, self._Logg_list, self._Z_list),
            np.arange(0, self._T_list.shape[0]))

    # def preload_phoenix_spectra(self):

    #     T_list = self.detect_all_T(self._phoenix_path)

    #     self._temperature_grid = np.array([x[0] for x in T_list])
    #     self.debug('Detected temepratures = %s',self._temperature_grid)
    #     self._avail_max_temp = max(self._temperature_grid)
    #     self._avail_min_temp = min(self._temperature_grid)

    #     self.debug('Temperature range = [%s-%s] ',self._avail_min_temp,self._avail_max_temp)
    #     self._max_index = self._temperature_grid.shape[0]
    #     self.spectra_grid = []

    #     #Load in all arrays
    #     for temp,f in T_list:
    #         self.debug('Loading %s %s',temp,f)
    #         arr = np.loadtxt(f)
    #         grid = 10000/np.copy(arr[:,0])
    #         sorted_idx = np.argsort(grid)

    #         self.wngrid = 10000/np.copy(arr[sorted_idx,0])
    #         self.spectra_grid.append(arr[sorted_idx,1]*10.0) #To SI

    # def find_closest_index(self,T):
    #     """
    #     Finds the two closest indices in our temeprature grid to our desired temperature

    #     Parameters
    #     ----------
    #     T: float
    #         Temperature in Kelvin

    #     Returns
    #     -------
    #     t_min: int
    #         Index to the left of ``T``

    #     t_max: int
    #         Index to the right of ``T``

    #     """

    #     t_min=self._temperature_grid.searchsorted(T,side='right')-1
    #     t_max = t_min+1

    #     return t_min,t_max

    # def interpolate_linear_temp(self,T):
    #     """
    #     Linearly interpolates the spectral emission density grid to the
    #     temperature given by ``T``

    #     Parameters
    #     ----------
    #     T: float
    #         Temeprature to interpolate to

    #     Returns
    #     -------
    #     out: :obj:`array`
    #         Spectral emission density interpolated to desired temperature

    #     """
    #     t_min,t_max = self.find_closest_index(T)
    #     if self._temperature_grid[t_min] == T:
    #         return self.spectra_grid[t_min]

    #     Tmax = self._temperature_grid[t_max]
    #     Tmin = self._temperature_grid[t_min]
    #     fx0=self.spectra_grid[t_min]
    #     fx1 = self.spectra_grid[t_max]

    #     return interp_lin_only(fx0,fx1,T,Tmin,Tmax)

    # def detect_all_T(self,path):
    #     """
    #     Finds files and detects all temperatures available in path

    #     Parameters
    #     ----------
    #     path: str
    #         Path to directory containing PHOENIX data

    #     """
    #     files = glob.glob(os.path.join(self._phoenix_path,'*.fits.gz'))
    #     files.sort()

    #     temp_list = []
    #     for f in files:
    #         #Gewt just the name of the file
    #         clean_name = pathlib.Path(f).stem
    #         #Split it by numbers
    #         split = re.split('(\d+)',clean_name)
    #         try:
    #             _T = float(split[1])
    #         except Exception:
    #             self.warning('Problem when reading filename %s',f)
    #             continue
    #         temp_list.append( (_T,f) )

    #     #Now sort the numbers
    #     temp_list.sort(key=lambda x: x[0])
    #     return temp_list

    def initialize(self, wngrid):
        """
        Initializes and interpolates the spectral emission density to the current
        stellar temperature and given wavenumber grid

        Parameters
        ----------
        wngrid: :obj:`array`
            Wavenumber grid to interpolate the SED to

        """
        # If temperature outside of range, use blavkbody
        if self.use_blackbody:
            self.warning('Using black body as temperature is outside of Star temeprature range {}'.format(
                self.temperature))
            super().initialize(wngrid)
        else:
            sed = self._base_sed
            self.sed = np.interp(wngrid, self.wngrid, sed)

    @property
    def spectralEmissionDensity(self):
        """
        Spectral emmision density

        Returns
        -------
        sed: :obj:`array`
        """
        return self.sed

    def write(self, output):
        star = super().write(output)
        star.write_string('phoenix_path', self._phoenix_path)
        return star

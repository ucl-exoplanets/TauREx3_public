from taurex.log import Logger
import numpy as np
from astropy.constants import N_A,k_B
import astropy.units as u
from functools import lru_cache
class RadisHITRANOpacity(Logger):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,molecule_name):
        super().__init__(self.__class__.__name__)
        import radis
        self.rad_xsec = radis.SpectrumFactory(wavenum_min=600,
                             wavenum_max=30000,
                             isotope='1',  #'all',
                             wstep=0.1,     # depends on HAPI benchmark.
                             verbose=0,
                             cutoff=1e-27,
                                        mole_fraction=1.0,
                             broadening_max_width=10.0,  # Corresponds to WavenumberWingHW/HWHM=50 in HAPI
                             molecule=molecule_name,
                             )
        self.rad_xsec.fetch_databank('astroquery',load_energies=False)

        self._molecule_name = molecule_name

        s = self.rad_xsec.eq_spectrum(Tgas=296, pressure=1)
        wn,absnce=s.get('abscoeff')

        self.wn = 1e7/wn
    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def wavenumberGrid(self):
        return self.wn

    @property
    def temperatureGrid(self):
        raise NotImplementedError
    
    @property
    def pressureGrid(self):
        raise NotImplementedError

    
    @lru_cache(maxsize=500)
    def compute_opacity(self,temperature,pressure):

        pressure_pascal = pressure*u.Pascal
        pressure_bar = pressure_pascal.to(u.bar)
        temperature_K = temperature *u.K

        density = ((pressure_bar)/(k_B*temperature_K)).value*1000

        s = self.rad_xsec.eq_spectrum(Tgas=temperature, pressure=pressure_bar.value)
        wn,absnce=s.get('abscoeff')

        return absnce/density

    



    def opacity(self,temperature,pressure,wngrid=None):
        orig=np.nan_to_num(self.compute_opacity(temperature,pressure))

        if wngrid is None:
            return orig
        else:
            # min_max =  (self.wavenumberGrid <= wngrid.max() ) & (self.wavenumberGrid >= wngrid.min())

            # total_bins = self.wavenumberGrid[min_max].shape[0]
            # if total_bins > wngrid.shape[0]:
            #     return np.append(np.histogram(self.wavenumberGrid,wngrid, weights=orig)[0]/np.histogram(self.wavenumberGrid,wngrid)[0],0)

            # else:
            return np.interp(wngrid,self.wavenumberGrid,orig)
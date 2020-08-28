from taurex.log import Logger
import numpy as np
from astropy.constants import N_A, k_B
import astropy.units as u
from functools import lru_cache


class RadisHITRANOpacity(Logger):
    """
    This is the base class for computing opactities

    """

    @classmethod
    def priority(cls):
        return 1000000


    @classmethod
    def discover(cls):
        from taurex.cache import GlobalCache

        if GlobalCache()['enable_radis'] is not True:
            return []

        trans = {'1': 'H2O',    '2': 'CO2',   '3': 'O3',      '4':  'N2O',   '5': 'CO',    '6': 'CH4',   '7': 'O2',     
            '9': 'SO2',   '10': 'NO2',  '11': 'NH3',    '12': 'HNO3', '13': 'OH', '14': 'HF',   '15': 'HCl',   '16': 'HBr',
            '17': 'HI',    '18': 'ClO',  '19': 'OCS',    '20': 'H2CO', '21': 'HOCl',    '23': 'HCN',   '24': 'CH3Cl',
            '25': 'H2O2',  '26': 'C2H2', '27': 'C2H6',   '28': 'PH3',  '29': 'COF2', '30': 'SF6',  '31': 'H2S',   '32': 'HCOOH',
            '33': 'HO2',   '34': 'O',    '35': 'ClONO2', '36': 'NO+',  '37': 'HOBr', '38': 'C2H4',  '40': 'CH3Br',
            '41': 'CH3CN', '42': 'CF4',  '43': 'C4H2',   '44': 'HC3N',   '46': 'CS',   '47': 'SO3'}
    
        mol_list = trans.values()
        wn_start, wn_end, wn_points = 600, 30000, 10000
        grid = GlobalCache()['radis_grid']
        if grid is not None:
            wn_start, wn_end, wn_points = grid


        return [(m, [m, wn_start, wn_end, wn_points]) for m in mol_list]




    def __init__(self, molecule_name, wn_start=600, wn_end=30000, wn_points=10000):
        super().__init__(self.__class__.__name__)
        import radis

        step = (wn_end-wn_start)/wn_points

        self.info('RADIS Grid set to %s %s %s', wn_start, wn_end, step)

        self.rad_xsec = radis.SpectrumFactory(wavenum_min=wn_start,
                                              wavenum_max=wn_end,
                                              isotope='1',  # 'all',
                                              # depends on HAPI benchmark.
                                              wstep=step,
                                              verbose=0,
                                              cutoff=1e-27,
                                              mole_fraction=1.0,
                                              broadening_max_width=10.0,  # Corresponds to WavenumberWingHW/HWHM=50 in HAPI
                                              molecule=molecule_name,
                                              )

        self.rad_xsec.fetch_databank('astroquery', load_energies=False)

        self._molecule_name = molecule_name

        s = self.rad_xsec.eq_spectrum(Tgas=296, pressure=1)
        wn, absnce = s.get('abscoeff')

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
    def compute_opacity(self, temperature, pressure):

        pressure_pascal = pressure*u.Pascal
        pressure_bar = pressure_pascal.to(u.bar)
        temperature_K = temperature * u.K

        density = ((pressure_bar)/(k_B*temperature_K)).value*1000

        s = self.rad_xsec.eq_spectrum(
            Tgas=temperature, pressure=pressure_bar.value)
        wn, absnce = s.get('abscoeff')

        return absnce/density

    def opacity(self, temperature, pressure, wngrid=None):
        orig = np.nan_to_num(self.compute_opacity(temperature, pressure))

        if wngrid is None:
            return orig
        else:
            # min_max =  (self.wavenumberGrid <= wngrid.max() ) & (self.wavenumberGrid >= wngrid.min())

            # total_bins = self.wavenumberGrid[min_max].shape[0]
            # if total_bins > wngrid.shape[0]:
            #     return np.append(np.histogram(self.wavenumberGrid,wngrid, weights=orig)[0]/np.histogram(self.wavenumberGrid,wngrid)[0],0)

            # else:
            return np.interp(wngrid, self.wavenumberGrid, orig)

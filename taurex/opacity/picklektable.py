from .interpolateopacity import InterpolatingOpacity
import pickle
import numpy as np
import pathlib


class PickleKTable(InterpolatingOpacity):
    """
    This is the base class for computing opactities
    """
    
    def __init__(self,filename,interpolation_mode='linear'):
        super().__init__('PickleKtable:{}'.format(pathlib.Path(filename).stem[0:10]),
                        interpolation_mode=interpolation_mode)

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._resolution = None
        self._load_pickle_file(filename)
        
        
    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def xsecGrid(self):
        return self._xsec_grid


    def _load_pickle_file(self, filename):

        #Load the pickle file
        self.info('Loading opacity from {}'.format(filename))
        try:
            with open(filename, 'rb') as f:
                self._spec_dict = pickle.load(f)
        except UnicodeDecodeError:
            with open(filename, 'rb') as f:
                self._spec_dict = pickle.load(f, encoding='latin1')       
        
        self._wavenumber_grid = self._spec_dict['bin_centers']
        self._ngauss = self._spec_dict['ngauss']
        self._temperature_grid = self._spec_dict['t']
        self._pressure_grid = self._spec_dict['p']*1e5
        self._xsec_grid = self._spec_dict['kcoeff']
        self._weights = self._spec_dict['weights']
        self._molecule_name = self._spec_dict['name']

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self.clean_molecule_name()

    def clean_molecule_name(self):
        splits = self.moleculeName.split('_')
        self._molecule_name = splits[0]

    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid
    
    @property
    def pressureGrid(self):
        return self._pressure_grid

    @property
    def resolution(self):
        return self._resolution

    @property
    def weights(self):
        return self._weights


        #return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))
    
    def opacity(self, temperature, pressure, wngrid=None):
        from scipy.interpolate import interp1d
        if wngrid is None:
            wngrid_filter = slice(None)
        else:
            wngrid_filter = np.where((self.wavenumberGrid >= wngrid.min()) & (
                self.wavenumberGrid <= wngrid.max()))[0]

        orig = self.compute_opacity(temperature, pressure, wngrid_filter).reshape(-1,len(self.weights))

        if wngrid is None or np.array_equal(self.wavenumberGrid.take(wngrid_filter), wngrid):
            return orig
        else:
            # min_max =  (self.wavenumberGrid <= wngrid.max() ) & (self.wavenumberGrid >= wngrid.min())

            # total_bins = self.wavenumberGrid[min_max].shape[0]
            # if total_bins > wngrid.shape[0]:
            #     return np.append(np.histogram(self.wavenumberGrid,wngrid, weights=orig)[0]/np.histogram(self.wavenumberGrid,wngrid)[0],0)

            # else:
            f = interp1d(self.wavenumberGrid[wngrid_filter], orig, axis=0, copy=False, bounds_error=False,fill_value=(orig[0],orig[-1]),assume_sorted=True)
            return f(wngrid)
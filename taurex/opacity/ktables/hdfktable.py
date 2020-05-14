from ..interpolateopacity import InterpolatingOpacity
import h5py
import numpy as np
import pathlib
from taurex.util.util import sanitize_molecule_string

class HDF5KTable(InterpolatingOpacity):
    """
    This is the base class for computing opactities using correlated k tables
    """
    
    def __init__(self, filename,interpolation_mode='linear', in_memory=True):
        self._molecule_name = sanitize_molecule_string(pathlib.Path(filename).stem.split('_')[0])
        super().__init__('HDF5Ktable:{}'.format(self._molecule_name),
                        interpolation_mode=interpolation_mode)
        self._molecule_name = sanitize_molecule_string(pathlib.Path(filename).stem.split('_')[0])
        self._filename = filename
        self._spec_dict = None

        self._resolution = None
        self.in_memory = in_memory
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

        self._spec_dict = h5py.File(filename, 'r')    
        
        self._wavenumber_grid = self._spec_dict['bin_centers'][...].astype(np.float64)
        self._ngauss = self._spec_dict['ngauss'][()]
        self._temperature_grid = self._spec_dict['t'][...].astype(np.float64)
        self._pressure_grid = self._spec_dict['p'][...].astype(np.float64)
        if self.in_memory:
            self._xsec_grid = self._spec_dict['kcoeff'][...].astype(np.float64)
        else:
            self._xsec_grid = self._spec_dict['kcoeff']
        self._weights = self._spec_dict['weights'][...].astype(np.float64)

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self.clean_molecule_name()
        if self.in_memory:
            self._spec_dict.close()

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
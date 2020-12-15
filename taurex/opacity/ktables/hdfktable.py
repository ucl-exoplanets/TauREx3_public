from ..interpolateopacity import InterpolatingOpacity
import h5py
import numpy as np
import pathlib
from taurex.util.util import sanitize_molecule_string
from .ktable import KTable


class HDF5KTable(KTable, InterpolatingOpacity):
    """
    This is the base class for computing opactities using correlated k tables
    """
    
    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        
        path = GlobalCache()['ktable_path']
        if path is None:
            return []
        path = os.path.join(path, '*.hdf5')

        files = glob.glob(path) + \
            glob.glob(os.path.join(GlobalCache()['ktable_path'], '*.h5'))

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'

        for f in files:
            splits = pathlib.Path(f).stem.split('_')
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery


    def __init__(self, filename, interpolation_mode='linear', in_memory=True):
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
        import astropy.units as u
        #Load the pickle file
        self.info('Loading opacity from {}'.format(filename))

        self._spec_dict = h5py.File(filename, 'r')    
        
        self._wavenumber_grid = self._spec_dict['bin_centers'][...].astype(np.float64)
        self._ngauss = self._spec_dict['ngauss'][()]
        self._temperature_grid = self._spec_dict['t'][...].astype(np.float64)

        pressure_units = self._spec_dict['p'].attrs['units']
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)


        self._pressure_grid = self._spec_dict['p'][...].astype(np.float64)*p_conversion



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

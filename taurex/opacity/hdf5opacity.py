from .interpolateopacity import InterpolatingOpacity
import pickle
import numpy as np
import pathlib


class HDF5Opacity(InterpolatingOpacity):
    """
    This is the base class for computing opactities

    """

    @classmethod
    def priority(cls):
        return 5

    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        from taurex.util.util import sanitize_molecule_string

        path = GlobalCache()['xsec_path']
        if path is None:
            return []
        path = [os.path.join(path, '*.h5'), os.path.join(path, '*.hdf5')]
        file_list = [f for glist in path for f in glob.glob(glist)]

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'
        mem = GlobalCache()['xsec_in_memory'] or True

        for f in file_list:
            op = HDF5Opacity(f, interpolation_mode='linear', in_memory=False)
            mol_name = op.moleculeName
            discovery.append((mol_name, [f, interp, mem]))
            #op._spec_dict.close()
            del op

        return discovery


    def __init__(self, filename, interpolation_mode='exp', in_memory=False):
        super().__init__('HDF5Opacity:{}'.format(pathlib.Path(filename).stem[0:10]),
                         interpolation_mode=interpolation_mode)

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self.in_memory = in_memory
        self._load_hdf_file(filename)

    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def xsecGrid(self):
        return self._xsec_grid

    def _load_hdf_file(self, filename):
        import h5py
        import astropy.units as u
        # Load the pickle file
        self.debug('Loading opacity from {}'.format(filename))

        self._spec_dict = h5py.File(filename, 'r')

        self._wavenumber_grid = self._spec_dict['bin_edges'][:]

        #temperature_units = self._spec_dict['t'].attrs['units']
        #t_conversion = u.Unit(temperature_units).to(u.K).value

        self._temperature_grid = self._spec_dict['t'][:]  # *t_conversion

        pressure_units = self._spec_dict['p'].attrs['units']
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)

        self._pressure_grid = self._spec_dict['p'][:]*p_conversion

        if self.in_memory:
            self._xsec_grid = self._spec_dict['xsecarr'][...]
        else:
            self._xsec_grid = self._spec_dict['xsecarr']

        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict['mol_name'][()]

        if isinstance(self._molecule_name, np.ndarray):
            self._molecule_name = self._molecule_name[0]
        
        try:
            self._molecule_name = self._molecule_name.decode()
        except (UnicodeDecodeError, AttributeError,):
            pass

        from taurex.util.util import ensure_string_utf8

        self._molecule_name = ensure_string_utf8(self._molecule_name)

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()

        if self.in_memory:
            self._spec_dict.close()
        

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

        # return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))

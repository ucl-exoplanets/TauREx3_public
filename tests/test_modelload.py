import unittest
from taurex.util.hdf5 import load_temperature_from_hdf5, load_pressure_from_hdf5
from taurex.output.hdf5 import HDF5Output
import tempfile,shutil
import numpy as np
import h5py


class HDFTester(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


    def gen_valid_hdf5_output(self,object_to_store,group_name):
        import os
        from taurex.output.hdf5 import HDF5Output

        from taurex.util.util import wnwidth_to_wlwidth, compute_bin_edges
        file_path = os.path.join(self.test_dir, 'test.hdf5')
        with HDF5Output(file_path) as f:
            group = f.create_group(group_name)
            object_to_store.write(group)
        
        return file_path


class TemperatureLoadTest(HDFTester):


    def test_isothermal(self):
        from taurex.data.profiles.temperature import Isothermal

        iso = Isothermal(T=600)

        file_path = self.gen_valid_hdf5_output(iso, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_temperature_from_hdf5(f['Test'])

        self.assertTrue(isinstance(loaded, Isothermal))
        self.assertEqual(iso.isoTemperature, loaded.isoTemperature)



    def test_guillot(self):
        from taurex.data.profiles.temperature import Guillot2010

        guil = Guillot2010(T_irr=600, kappa_v1=0.5)

        file_path = self.gen_valid_hdf5_output(guil, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_temperature_from_hdf5(f['Test'])

        self.assertTrue(isinstance(loaded, Guillot2010))
        self.assertEqual(guil.equilTemperature, loaded.equilTemperature)
        self.assertEqual(guil.kappa_ir, loaded.kappa_ir)
        self.assertEqual(guil.kappa_v1, loaded.kappa_v1)
        self.assertEqual(guil.kappa_v2, loaded.kappa_v2)
        self.assertEqual(guil.alpha, loaded.alpha)

    def test_2point(self):
        from taurex.data.profiles.temperature import NPoint
        twop = NPoint(T_surface=200.0, T_top=300.0, P_surface=-1, P_top=1e-6)
        
        file_path = self.gen_valid_hdf5_output(twop, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_temperature_from_hdf5(f['Test'])

        self.assertTrue(isinstance(loaded, NPoint))

        self.assertEqual(twop.temperatureSurface, loaded.temperatureSurface)
        self.assertEqual(twop.temperatureTop, loaded.temperatureTop)
        self.assertEqual(twop.pressureSurface, loaded.pressureSurface)
        self.assertEqual(twop.pressureTop, loaded.pressureTop)

    def test_3point(self):
        from taurex.data.profiles.temperature import NPoint
        threep = NPoint(T_surface=200.0, T_top=300.0, P_surface=-1,
                        temperature_points=[150.0],
                        pressure_points=[1e2],
                        P_top=1e-6)

        file_path = self.gen_valid_hdf5_output(threep, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_temperature_from_hdf5(f['Test'])

        self.assertTrue(isinstance(loaded, NPoint))

        self.assertEqual(threep.temperatureSurface, loaded.temperatureSurface)
        self.assertEqual(threep.temperatureTop, loaded.temperatureTop)
        self.assertEqual(threep.pressureSurface, loaded.pressureSurface)
        self.assertEqual(threep.pressureTop, loaded.pressureTop)
        self.assertEqual(len(threep._t_points), len(loaded._t_points))
        self.assertEqual(len(threep._p_points), len(loaded._p_points))


class PressLoadTest(HDFTester):

    def test_simple_pressure(self):
        from taurex.data.profiles.pressure import SimplePressureProfile

        pres = SimplePressureProfile(nlayers=50, atm_min_pressure=1e-5,
                                     atm_max_pressure=1e5)
        pres.compute_pressure_profile()
        file_path = self.gen_valid_hdf5_output(pres, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_pressure_from_hdf5(f['Test'])

        self.assertTrue(isinstance(loaded, SimplePressureProfile))
        self.assertEqual(pres.nLayers, loaded.nLayers)
        
        loaded.compute_pressure_profile()
        np.testing.assert_array_equal(pres.profile, loaded.profile)

import unittest
from taurex.util.hdf5 import load_temperature_from_hdf5, load_pressure_from_hdf5, load_gas_from_hdf5, load_chemistry_from_hdf5
from taurex.output.hdf5 import HDF5Output
import tempfile
import shutil
import numpy as np
import h5py
from unittest.mock import patch


class HDFTester(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def gen_valid_hdf5_output(self, object_to_store, group_name):
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


class GasLoadTest(HDFTester):

    def test_constant_gas(self):
        from taurex.data.profiles.chemistry import ConstantGas

        gas = ConstantGas('H2O', 1e-4)
        file_path = self.gen_valid_hdf5_output(gas, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_gas_from_hdf5(f['Test'], 'H2O')

        self.assertTrue(isinstance(loaded, ConstantGas))
        self.assertEqual(gas._mix_ratio, loaded._mix_ratio)

    def test_twolayer_gas(self):
        from taurex.data.profiles.chemistry import TwoLayerGas

        gas = TwoLayerGas('H2O', 1e-4, 1e-7, 1e2, 15)
        file_path = self.gen_valid_hdf5_output(gas, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_gas_from_hdf5(f['Test'], 'H2O')

        self.assertTrue(isinstance(loaded, TwoLayerGas))
        self.assertEqual(gas._mix_ratio_pressure, loaded._mix_ratio_pressure)
        self.assertEqual(gas._mix_surface, loaded._mix_surface)
        self.assertEqual(gas._mix_top, loaded._mix_top)
        self.assertEqual(gas._mix_ratio_smoothing, loaded._mix_ratio_smoothing)


class PlanetLoadTest(HDFTester):

    def test_planet(self):
        from taurex.data.planet import Planet
        from taurex.util.hdf5 import load_planet_from_hdf5

        planet = Planet(1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1000.0)

        file_path = self.gen_valid_hdf5_output(planet, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_planet_from_hdf5(f['Test'])

        self.assertTrue(isinstance(loaded, Planet))
        self.assertEqual(planet._mass, loaded._mass)
        self.assertEqual(planet._radius, loaded._radius)
        self.assertEqual(planet._transit_time, loaded._transit_time)
        self.assertEqual(planet._distance, loaded._distance)
        self.assertEqual(planet._orbit_period, loaded._orbit_period)


class StarLoadTest(HDFTester):

    def test_blackbody(self):
        from taurex.data.stellar import BlackbodyStar
        from taurex.util.hdf5 import load_star_from_hdf5

        star = BlackbodyStar(6000, 1.5, 1.5, 15.0, 1.5, 1.5)
        wngrid = np.linspace(300, 3000, 10)
        star.initialize(wngrid)
        file_path = self.gen_valid_hdf5_output(star, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_star_from_hdf5(f['Test'])
        loaded.initialize(wngrid)
        self.assertTrue(isinstance(loaded, BlackbodyStar))
        self.assertEqual(star._mass, loaded._mass)
        self.assertEqual(star._metallicity, loaded._metallicity)
        self.assertEqual(star._radius, loaded._radius)
        self.assertEqual(star.distance, loaded.distance)
        self.assertEqual(star.magnitudeK, loaded.magnitudeK)
        self.assertEqual(star.temperature, loaded.temperature)
        np.testing.assert_array_equal(star.sed, loaded.sed)


class ChemistryLoadTest(HDFTester):

    def test_taurex_chemistry(self):
        from taurex.data.profiles.chemistry import TaurexChemistry, ConstantGas
        from taurex.cache import OpacityCache

        molecules = ['H2O', 'CH4']
        mix_ratios = [1e-2, 1e-8]
        molecule_classes = [ConstantGas, ConstantGas]

        with patch.object(OpacityCache, "find_list_of_molecules") as mock_my_method:
            mock_my_method.return_value = molecules

            chemistry = TaurexChemistry(fill_gases=['H2', 'N2'], ratio=0.145)

            for mol, mix, klass in zip(molecules, mix_ratios, molecule_classes):
                chemistry.addGas(klass(mol, mix))

        chemistry.initialize_chemistry(100, None, None, None)
        file_path = self.gen_valid_hdf5_output(chemistry, 'Test')

        with patch.object(OpacityCache, "find_list_of_molecules") as mock_my_method:
            mock_my_method.return_value = molecules
            with h5py.File(file_path, 'r') as f:
                loaded = load_chemistry_from_hdf5(f['Test'])

        loaded.initialize_chemistry(100, None, None, None)
        self.assertTrue(set(chemistry._fill_gases) == set(loaded._fill_gases))
        self.assertTrue(set(chemistry._fill_ratio) == set(loaded._fill_ratio))
        np.testing.assert_equal(
            chemistry.activeGasMixProfile, loaded.activeGasMixProfile)


class ContribLoadTest(HDFTester):

    def test_absorption(self):
        from taurex.contributions import AbsorptionContribution
        from taurex.util.hdf5 import load_contrib_from_hdf5
        abs_c = AbsorptionContribution()
        file_path = self.gen_valid_hdf5_output(abs_c, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_contrib_from_hdf5(
                f['Test'], AbsorptionContribution.__name__)

        self.assertTrue(isinstance(loaded, AbsorptionContribution))

    def test_rayleigh(self):
        from taurex.contributions import RayleighContribution
        from taurex.util.hdf5 import load_contrib_from_hdf5
        ray_c = RayleighContribution()
        file_path = self.gen_valid_hdf5_output(ray_c, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_contrib_from_hdf5(
                f['Test'], RayleighContribution.__name__)

        self.assertTrue(isinstance(loaded, RayleighContribution))

    def test_cia(self):
        from taurex.contributions import CIAContribution
        from taurex.util.hdf5 import load_contrib_from_hdf5
        cia_c = CIAContribution(cia_pairs=['H2-He', 'H2-H2'])
        file_path = self.gen_valid_hdf5_output(cia_c, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_contrib_from_hdf5(f['Test'],
                                            CIAContribution.__name__)

        self.assertTrue(isinstance(loaded, CIAContribution))
        self.assertTrue(set(cia_c._cia_pairs) == set(loaded._cia_pairs))

    def test_simpleclouds(self):
        from taurex.contributions import SimpleCloudsContribution
        from taurex.util.hdf5 import load_contrib_from_hdf5
        clouds_c = SimpleCloudsContribution(clouds_pressure=1e3)
        file_path = self.gen_valid_hdf5_output(clouds_c, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_contrib_from_hdf5(f['Test'],
                                            SimpleCloudsContribution.__name__)

        self.assertTrue(isinstance(loaded, SimpleCloudsContribution))
        self.assertEqual(clouds_c.cloudsPressure, loaded.cloudsPressure)

    def test_leemie(self):
        from taurex.contributions import LeeMieContribution
        from taurex.util.hdf5 import load_contrib_from_hdf5
        lee_mie = LeeMieContribution(1.0, 20, 1e-4, 1e-2, 1e-3)
        file_path = self.gen_valid_hdf5_output(lee_mie, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_contrib_from_hdf5(f['Test'],
                                            LeeMieContribution.__name__)

        self.assertTrue(isinstance(loaded, LeeMieContribution))
        self.assertEqual(lee_mie.mieQ, loaded.mieQ)
        self.assertEqual(lee_mie.mieRadius, loaded.mieRadius)
        self.assertEqual(lee_mie.mieBottomPressure, loaded.mieBottomPressure)
        self.assertEqual(lee_mie.mieTopPressure, loaded.mieTopPressure)
        self.assertEqual(lee_mie.mieMixing, loaded.mieMixing)


class ModelLoadTest(HDFTester):

    def test_transmission(self):
        from taurex.util.hdf5 import load_model_from_hdf5
        from taurex.model import TransmissionModel
        from taurex.data.profiles.chemistry import TaurexChemistry, ConstantGas
        from taurex.contributions import AbsorptionContribution, RayleighContribution
        from taurex.cache import OpacityCache

        with patch.object(OpacityCache, "find_list_of_molecules") as mock_my_method:
            mock_my_method.return_value = ['H2O']
            chem = TaurexChemistry()
            chem.addGas(ConstantGas())

        tm = TransmissionModel(chemistry=chem)
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(RayleighContribution())
        tm.build()
        tm.initialize_profiles()
        wngrid = np.linspace(100, 400)
        tm.star.initialize(wngrid)
        with patch.object(TransmissionModel, "model") as mock_my_method:
            mock_my_method = None
            file_path = self.gen_valid_hdf5_output(tm, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_model_from_hdf5(f['Test']['ModelParameters'])

        loaded.build()
        loaded.initialize_profiles()
        wngrid = np.linspace(100, 400)

        self.assertTrue(isinstance(loaded, TransmissionModel))
        np.testing.assert_array_equal(loaded.densityProfile, tm.densityProfile)

        truth_contrib = [type(c) for c in tm.contribution_list]
        loaded_contrib = [type(c) for c in loaded.contribution_list]

        self.assertTrue(set(truth_contrib) == set(loaded_contrib))

    def test_emission(self):
        from taurex.util.hdf5 import load_model_from_hdf5
        from taurex.model import EmissionModel
        from taurex.data.profiles.chemistry import TaurexChemistry, ConstantGas
        from taurex.contributions import AbsorptionContribution, RayleighContribution
        from taurex.cache import OpacityCache

        with patch.object(OpacityCache, "find_list_of_molecules") as mock_my_method:
            mock_my_method.return_value = ['H2O']
            chem = TaurexChemistry()
            chem.addGas(ConstantGas())

        tm = EmissionModel(chemistry=chem, ngauss=10)
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(RayleighContribution())
        tm.build()
        tm.initialize_profiles()
        wngrid = np.linspace(100, 400)
        tm.star.initialize(wngrid)
        with patch.object(EmissionModel, "model") as mock_my_method:
            mock_my_method = None
            file_path = self.gen_valid_hdf5_output(tm, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_model_from_hdf5(f['Test']['ModelParameters'])

        loaded.build()
        loaded.initialize_profiles()
        wngrid = np.linspace(100, 400)

        self.assertTrue(isinstance(loaded, EmissionModel))
        np.testing.assert_array_equal(loaded.densityProfile, tm.densityProfile)

        truth_contrib = [type(c) for c in tm.contribution_list]
        loaded_contrib = [type(c) for c in loaded.contribution_list]

        self.assertTrue(set(truth_contrib) == set(loaded_contrib))
        self.assertEqual(tm._ngauss, loaded._ngauss)

    def test_directimage(self):
        from taurex.util.hdf5 import load_model_from_hdf5
        from taurex.model import DirectImageModel
        from taurex.data.profiles.chemistry import TaurexChemistry, ConstantGas
        from taurex.contributions import AbsorptionContribution, RayleighContribution
        from taurex.cache import OpacityCache

        with patch.object(OpacityCache, "find_list_of_molecules") as mock_my_method:
            mock_my_method.return_value = ['H2O']
            chem = TaurexChemistry()
            chem.addGas(ConstantGas())

        tm = DirectImageModel(chemistry=chem, ngauss=10)
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(RayleighContribution())
        tm.build()
        tm.initialize_profiles()
        wngrid = np.linspace(100, 400)
        tm.star.initialize(wngrid)
        with patch.object(DirectImageModel, "model") as mock_my_method:
            mock_my_method = None
            file_path = self.gen_valid_hdf5_output(tm, 'Test')

        with h5py.File(file_path, 'r') as f:
            loaded = load_model_from_hdf5(f['Test']['ModelParameters'])

        loaded.build()
        loaded.initialize_profiles()
        wngrid = np.linspace(100, 400)

        self.assertTrue(isinstance(loaded, DirectImageModel))
        np.testing.assert_array_equal(loaded.densityProfile, tm.densityProfile)

        truth_contrib = [type(c) for c in tm.contribution_list]
        loaded_contrib = [type(c) for c in loaded.contribution_list]

        self.assertTrue(set(truth_contrib) == set(loaded_contrib))
        self.assertEqual(tm._ngauss, loaded._ngauss)

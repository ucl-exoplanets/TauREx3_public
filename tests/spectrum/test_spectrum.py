import unittest
from unittest.mock import patch
from taurex.data.spectrum.spectrum import BaseSpectrum
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.data.spectrum.taurex import TaurexSpectrum
import numpy as np
import tempfile
import shutil


test_data_with_bin = np.array([[7.27555920e+00, 1.45008637e-02, 7.42433558e-05, 4.76957670e-01],
                               [6.81373911e+00, 1.45861957e-02,
                                   7.06735254e-05, 4.46682521e-01],
                               [6.38123330e+00, 1.43091103e-02,
                                   6.47629305e-05, 4.18329104e-01],
                               [5.97618103e+00, 1.43232253e-02,
                                   6.07624822e-05, 3.91775435e-01],
                               [5.59683967e+00, 1.44375945e-02,
                                   6.01144320e-05, 3.66907276e-01],
                               [5.24157722e+00, 1.44452048e-02,
                                   5.75032500e-05, 3.43617636e-01],
                               [4.90886524e+00, 1.45200834e-02,
                                   5.23495218e-05, 3.21806319e-01],
                               [4.59727234e+00, 1.44954295e-02,
                                   5.14375698e-05, 3.01379487e-01],
                               [4.30545796e+00, 1.41630681e-02,
                                   4.79325110e-05, 2.82249259e-01],
                               [4.03216667e+00, 1.38851464e-02,
                                   4.55689869e-05, 2.64333333e-01],
                               [3.72344897e+00, 1.38783299e-02,
                                   7.09940727e-05, 7.40966530e-02],
                               [3.65008232e+00, 1.39306124e-02,
                                   6.75721978e-05, 7.26366562e-02],
                               [3.57816128e+00, 1.39690888e-02,
                                   6.71354036e-05, 7.12054271e-02],
                               [3.50765736e+00, 1.40176284e-02,
                                   6.62062054e-05, 6.98023989e-02],
                               [3.43854266e+00, 1.40220670e-02,
                                   6.70048563e-05, 6.84270159e-02],
                               [3.37078978e+00, 1.41101703e-02,
                                   6.47573583e-05, 6.70787333e-02],
                               [3.30437191e+00, 1.41335711e-02,
                                   6.24627939e-05, 6.57570173e-02],
                               [3.23926272e+00, 1.42214076e-02,
                                   6.08163087e-05, 6.44613443e-02],
                               [3.17543645e+00, 1.42792404e-02,
                                   5.89126227e-05, 6.31912011e-02],
                               [3.11286781e+00, 1.43024513e-02,
                                   5.83923231e-05, 6.19460848e-02],
                               [3.05153202e+00, 1.43843733e-02,
                                   5.85608968e-05, 6.07255022e-02],
                               [2.99140478e+00, 1.44130864e-02,
                                   5.63454197e-05, 5.95289699e-02],
                               [2.93246229e+00, 1.44080243e-02,
                                   5.52744519e-05, 5.83560140e-02],
                               [2.87468120e+00, 1.44070635e-02,
                                   5.41598358e-05, 5.72061700e-02],
                               [2.81803862e+00, 1.44043868e-02,
                                   5.45431137e-05, 5.60789825e-02],
                               [2.76251213e+00, 1.44350381e-02,
                                   5.45881312e-05, 5.49740050e-02],
                               [2.70807972e+00, 1.43717219e-02,
                                   5.46566154e-05, 5.38907999e-02],
                               [2.65471985e+00, 1.43276262e-02,
                                   5.20714025e-05, 5.28289382e-02],
                               [2.60241139e+00, 1.43253806e-02,
                                   5.13106598e-05, 5.17879994e-02],
                               [2.55113360e+00, 1.43732044e-02,
                                   5.10290621e-05, 5.07675713e-02],
                               [2.50086619e+00, 1.44053308e-02,
                                   5.24233842e-05, 4.97672495e-02],
                               [2.45158925e+00, 1.43047985e-02,
                                   5.10704065e-05, 4.87866381e-02],
                               [2.40328325e+00, 1.42253121e-02,
                                   4.89367922e-05, 4.78253486e-02],
                               [2.35592908e+00, 1.41272315e-02,
                                   4.84612973e-05, 4.68830003e-02],
                               [2.30950797e+00, 1.40818772e-02,
                                   4.88309653e-05, 4.59592200e-02],
                               [2.26400154e+00, 1.39034460e-02,
                                   4.78506157e-05, 4.50536418e-02],
                               [2.21939176e+00, 1.39024280e-02,
                                   4.62754271e-05, 4.41659071e-02],
                               [2.17566098e+00, 1.39000354e-02,
                                   4.61949457e-05, 4.32956642e-02],
                               [2.13279186e+00, 1.39305842e-02,
                                   4.63551039e-05, 4.24425686e-02],
                               [2.09076743e+00, 1.40054149e-02,
                                   4.46124318e-05, 4.16062823e-02],
                               [2.04957106e+00, 1.40649314e-02,
                                   4.45070081e-05, 4.07864742e-02],
                               [2.00918641e+00, 1.41145201e-02,
                                   4.93795583e-05, 3.99828195e-02],
                               [1.96959750e+00, 1.41361402e-02,
                                   4.78705968e-05, 3.91950000e-02],
                               [1.88361302e+00, 1.40971503e-02,
                                   2.79199901e-05, 1.83657869e-01],
                               [1.70849254e+00, 1.38297816e-02,
                                   2.42975533e-05, 1.66583101e-01],
                               [1.54965310e+00, 1.38920059e-02,
                                   2.36878254e-05, 1.51095783e-01],
                               [1.40558104e+00, 1.39972918e-02,
                                   2.35131391e-05, 1.37048330e-01],
                               [1.27490344e+00, 1.35321842e-02,
                                   2.36384144e-05, 1.24306875e-01],
                               [1.15637500e+00, 1.36119825e-02,
                                   2.39933743e-05, 1.12750000e-01],
                               [9.55000000e-01, 1.34500037e-02,
                                   2.13757755e-05, 2.90000000e-01],
                               [7.05000000e-01, 1.35609008e-02,
                                   2.16687213e-05, 2.10000000e-01],
                               [5.50000000e-01, 1.37534564e-02, 2.27756279e-05, 1.00000000e-01]])

test_data_without_bin = test_data_with_bin[:, 0:3]


class BaseSpectrumTest(unittest.TestCase):

    def test_properties(self):
        spec = BaseSpectrum('test')

        with self.assertRaises(NotImplementedError):
            spec.spectrum

        with self.assertRaises(NotImplementedError):
            spec.wavelengthGrid

        with self.assertRaises(NotImplementedError):
            spec.wavenumberGrid

        with self.assertRaises(NotImplementedError):
            spec.binWidths

        with self.assertRaises(NotImplementedError):
            spec.binEdges


class ObservedSpectrumTest(unittest.TestCase):

    def _taurex_binwidths(self):
        # bins given as input
        obs_binwidths = test_data_with_bin[:, 3]
        obs_wlgrid = test_data_with_bin[:, 0]
        # calculate bin edges
        obs_wlgrid_ascending = obs_wlgrid[::-1]
        obs_binwidths_ascending = obs_binwidths[::-1]
        bin_edges = []
        for i in range(len(obs_wlgrid)):
            bin_edges.append(
                obs_wlgrid_ascending[i]-obs_binwidths_ascending[i]/2.)
            bin_edges.append(
                obs_wlgrid_ascending[i]+obs_binwidths_ascending[i]/2.)
        obs_binedges = bin_edges[::-1]

        return np.array(obs_binedges)

    def _taurex_extrap_bin(self):

        obs_wlgrid = test_data_without_bin[:, 0]
        bin_edges = []
        # first bin edge
        bin_edges.append(obs_wlgrid[0] - (obs_wlgrid[1]-obs_wlgrid[0])/2.0)
        for i in range(len(obs_wlgrid)-1):
            bin_edges.append(obs_wlgrid[i]+(obs_wlgrid[i+1]-obs_wlgrid[i])/2.0)
        # last bin edge
        bin_edges.append((obs_wlgrid[-1]-obs_wlgrid[-2])/2.0 + obs_wlgrid[-1])
        bin_widths = np.abs(np.diff(bin_edges))

        return bin_widths, bin_edges

    @patch("numpy.loadtxt", return_value=np.ones(shape=(1000, 4), dtype=np.float))
    def test_with_bin(self, mock_load):
        test_spec = ObservedSpectrum('TestFile')
        mock_load.assert_called_with('TestFile')
        test_spec.spectrum

        test_wnumber = np.zeros(1000)
        test_wnumber[...] = 10000

        np.testing.assert_array_equal(test_spec.wavelengthGrid, np.ones(1000))
        np.testing.assert_array_equal(test_spec.wavenumberGrid, test_wnumber)
        np.testing.assert_array_equal(test_spec.spectrum, np.ones(1000))
        np.testing.assert_array_equal(test_spec.errorBar, np.ones(1000))

        self.assertIsNotNone(test_spec.binEdges)
        self.assertIsNotNone(test_spec.binWidths)

    @patch("numpy.loadtxt", return_value=np.ones(shape=(1000, 3), dtype=np.float))
    def test_without_bin(self, mock_load):
        test_spec = ObservedSpectrum('TestFileNoBin')
        mock_load.assert_called_with('TestFileNoBin')
        test_spec.spectrum

        test_wnumber = np.zeros(1000)
        test_wnumber[...] = 10000

        np.testing.assert_array_equal(test_spec.wavelengthGrid, np.ones(1000))
        np.testing.assert_array_equal(test_spec.wavenumberGrid, test_wnumber)
        np.testing.assert_array_equal(test_spec.spectrum, np.ones(1000))
        np.testing.assert_array_equal(test_spec.errorBar, np.ones(1000))

        self.assertIsNotNone(test_spec.binEdges)
        self.assertIsNotNone(test_spec.binWidths)


class TaurexSpectrumTest(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def gen_valid_hdf5_output(self):
        import os
        from taurex.output.hdf5 import HDF5Output

        from taurex.util.util import wnwidth_to_wlwidth, compute_bin_edges
        file_path = os.path.join(self.test_dir, 'test.hdf5')

        test_dict = {}

        wngrid = np.linspace(100, 1000, 100)
        wlgrid = 10000/wngrid
        spectrum = np.random.rand(100)
        error = np.random.rand(100)
        wnwidth = compute_bin_edges(wngrid)[-1]
        wlwidth = wnwidth_to_wlwidth(wngrid, wnwidth)

        test_dict['instrument_wlgrid'] = wlgrid
        test_dict['instrument_wngrid'] = wngrid
        test_dict['instrument_spectrum'] = spectrum
        test_dict['instrument_noise'] = error
        test_dict['instrument_wnwidth'] = wnwidth

        with HDF5Output(file_path) as f:

            group = f.create_group('Output')
            group.store_dictionary(test_dict, group_name='Spectra')

        return file_path, wngrid, wlgrid, spectrum, error, wnwidth, wlwidth

    def test_valid_opt(self):

        res = self.gen_valid_hdf5_output()
        file_path, wngrid, wlgrid, spectrum, error, wnwidth, wlwidth = res
        ts = TaurexSpectrum(file_path)
        self.assertEqual(ts._obs_spectrum.shape[0], 100)
        self.assertEqual(ts._obs_spectrum.shape[1], 4)
        np.testing.assert_array_equal(ts.spectrum, spectrum)
        np.testing.assert_array_equal(ts.wavelengthGrid, wlgrid)
        np.testing.assert_array_equal(ts.errorBar, error)
        np.testing.assert_array_almost_equal(ts.binWidths, wnwidth)

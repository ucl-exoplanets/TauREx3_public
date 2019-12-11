import unittest
import numpy as np

from taurex.binning.binner import Binner
from taurex.binning.simplebinner import SimpleBinner
from taurex.binning.fluxbinner import FluxBinner


class BinnerTest(unittest.TestCase):

    def test_exception(self):

        binner = Binner()

        with self.assertRaises(NotImplementedError):
            binner.bindown(None, None)


class SimpleBinnerTest(unittest.TestCase):

    def test_simple_binning_1D(self):

        wngrid = np.linspace(10, 100, 100)

        bingrid = np.linspace(4, 8, 10)

        data = np.arange(100)

        sb = SimpleBinner(bingrid)

        result = sb.bindown(wngrid, data)[1]

        self.assertEqual(bingrid.shape, result.shape)

    def test_simple_binning_2D(self):

        wngrid = np.linspace(10, 100, 100)

        bingrid = np.linspace(4, 8, 10)

        data = np.arange(1000).reshape(10, 100)

        sb = SimpleBinner(bingrid)

        result = sb.bindown(wngrid, data)[1]

        self.assertEqual(bingrid.shape[-1], result.shape[-1])
        self.assertEqual(data.shape[0], result.shape[0])


class FluxBinnerTest(unittest.TestCase):

    def test_init(self):

        fb = FluxBinner(np.linspace(300.0, 30000.0, 500), np.ones(500)*0.5)
        fb = FluxBinner(np.linspace(300.0, 30000.0, 500), 0.5)

        np.testing.assert_equal(fb._wngrid_width, np.ones(500)*0.5)

        fb = FluxBinner(np.linspace(300.0, 30000.0, 500))

        with self.assertRaises(ValueError):
            fb = FluxBinner(np.linspace(300.0, 30000.0, 500), np.ones(450)*0.5)

    def test_binning(self):

        fake_bins = np.array([[1.5, 0.1], [3.0, 0.1], [4.5, 0.1]])
        spaced_fake_bins = np.linspace(1.0, 5.0, int(4/0.1)+1)

        fake_spectra = np.random.rand(3000).reshape(1000, 3)

        fake_spectra[:, 0] = np.linspace(0.5, 50, 1000)

        fb_small = FluxBinner(fake_bins[:, 0], fake_bins[:, 1])
        fb_big = FluxBinner(spaced_fake_bins[:], 0.1)

        res_small = fb_small.bindown(fake_spectra[:, 0], fake_spectra[:, 1],
                                     error=fake_spectra[:, 2])
        res_big = fb_big.bindown(fake_spectra[:, 0], fake_spectra[:, 1],
                                 error=fake_spectra[:, 2])

        for val in res_small[1]:
            self.assertIn(val, res_big[1])

    def test_binning_2d(self):

        fake_bins = np.array([[1.5, 0.1], [3.0, 0.1], [4.5, 0.1]])
        spaced_fake_bins = np.linspace(1.0, 5.0, int(4/0.1)+1)

        fake_spectra = np.random.rand(3000).reshape(1000, 3)

        fake_spectra[:, 0] = np.linspace(0.5, 50, 1000)

        spectra2d = np.random.rand(2000).reshape(2, 1000)

        fb_small = FluxBinner(fake_bins[:, 0], fake_bins[:, 1])

        res = fb_small.bindown(fake_spectra[:, 0], spectra2d)

        self.assertEqual(res[1].shape[0], 2)
        self.assertEqual(res[1].shape[-1], fake_bins.shape[0])

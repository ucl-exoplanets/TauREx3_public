import unittest
import numpy as np
from taurex.binning.binner import Binner
from taurex.binning.simplebinner import SimpleBinner


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

        result = sb.bindown(wngrid, data)

        self.assertEqual(bingrid.shape, result.shape)

    def test_simple_binning_2D(self):

        wngrid = np.linspace(10, 100, 100)

        bingrid = np.linspace(4, 8, 10)

        data = np.arange(1000).reshape(10, 100)

        sb = SimpleBinner(bingrid)

        result = sb.bindown(wngrid, data)

        self.assertEqual(bingrid.shape[-1], result.shape[-1])

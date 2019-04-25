import unittest
from unittest.mock import patch
from taurex.data.spectrum.spectrum import BaseSpectrum
from taurex.data.spectrum.observed import ObservedSpectrum
import numpy as np
 
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
    


    @patch("numpy.loadtxt", return_value=np.ones(shape=(1000,4),dtype=np.float))
    def test_with_bin(self,mock_load):
        test_spec = ObservedSpectrum('TestFile')
        mock_load.assert_called_with('TestFile')
        test_spec.spectrum

        test_wnumber = np.zeros(1000)
        test_wnumber[...] = 10000        

        np.testing.assert_array_equal(test_spec.wavelengthGrid,np.ones(1000))
        np.testing.assert_array_equal(test_spec.wavenumberGrid,test_wnumber)
        np.testing.assert_array_equal(test_spec.spectrum,np.ones(1000))
        np.testing.assert_array_equal(test_spec.errorBar,np.ones(1000))

        self.assertIsNotNone(test_spec.binEdges)
        self.assertIsNotNone(test_spec.binWidths)

        
    @patch("numpy.loadtxt", return_value=np.ones(shape=(1000,3),dtype=np.float))
    def test_without_bin(self,mock_load):
        test_spec = ObservedSpectrum('TestFileNoBin')
        mock_load.assert_called_with('TestFileNoBin')
        test_spec.spectrum

        test_wnumber = np.zeros(1000)
        test_wnumber[...] = 10000        

        np.testing.assert_array_equal(test_spec.wavelengthGrid,np.ones(1000))
        np.testing.assert_array_equal(test_spec.wavenumberGrid,test_wnumber)
        np.testing.assert_array_equal(test_spec.spectrum,np.ones(1000))
        np.testing.assert_array_equal(test_spec.errorBar,np.ones(1000))
        
        self.assertIsNotNone(test_spec.binEdges)
        self.assertIsNotNone(test_spec.binWidths)

        
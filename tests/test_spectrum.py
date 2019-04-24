import unittest
import numpy as np
from taurex.data.spectrum.spectrum import BaseSpectrum
 
 
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
    pass
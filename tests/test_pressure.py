import unittest
import numpy as np
from taurex.data.profiles.pressureprofiles.pressureprofile import PressureProfile,TaurexPressureProfile


class PressureProfileTest(unittest.TestCase):


    def test_not_working(self):

        pp = PressureProfile('test',100)

        with self.assertRaises(NotImplementedError):
            pp.compute_pressure_profile()
        
        with self.assertRaises(NotImplementedError):
            pp.profile


class TaurexPressureProfileTest(unittest.TestCase):


    def test_working(self):

        pp = TaurexPressureProfile(100,0.1,1000)
        pp.compute_pressure_profile()
        
        pp.profile

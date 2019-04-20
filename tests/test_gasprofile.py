import unittest
import numpy as np
from taurex.data.profiles.gasprofiles.gasprofile import GasProfile


class GasProfileTest(unittest.TestCase):
 
    def setUp(self):
 
        self.tp = GasProfile('test')
    
        #test_layers = 10

        #pres_prof = np.ones(test_layers)

        #self.tp.initialize_profile(10,pres_prof,pres_prof,pres_prof)




    def test_compute_active(self):
        with self.assertRaises(NotImplementedError):
            self.tp.compute_active_gas_profile()
    
    def test_compute_inactive(self):
        with self.assertRaises(NotImplementedError):
            self.tp.compute_inactive_gas_profile()
    
    def test_get_profiles(self):
        self.assertEqual(self.tp.activeGasMixProfile,None)
        self.assertEqual(self.tp.inActiveGasMixProfile,None)

        
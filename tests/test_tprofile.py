import unittest
import numpy as np
from taurex.data.profiles.temperature.tprofile import TemperatureProfile
 
 
 
class TemperatureProfileTest(unittest.TestCase):
 
    def setUp(self):
 
        self.tp = TemperatureProfile('test')
 
         
 
    def test_init(self):
        
        test_layers = 10

        pres_prof = np.ones(test_layers)

        self.tp.initialize_profile(None,10,pres_prof)

    def test_unimplemeted(self):
        with self.assertRaises(NotImplementedError):
            self.tp.profile


class IsoThermalTest(unittest.TestCase):
 
    def setUp(self):
        from taurex.data.profiles.temperature.isothermal import Isothermal
        self.tp = Isothermal(100.0)
        test_layers = 10

        pres_prof = np.ones(test_layers)

        self.tp.initialize_profile(None,10,pres_prof)
         
 
    def test_compute_profile(self):
        self.tp.profile


class GuillotTest(unittest.TestCase):
 

    tp_guillot_T_irr = 1500
    tp_guillot_kappa_ir = 0.5
    tp_guillot_kappa_v1 = 0.05
    tp_guillot_kappa_v2 = 0.05
    tp_guillot_alpha = 0.005

    def setUp(self):
        from taurex.data.profiles.temperature.guillot import Guillot2010
        from taurex.data.planet import Earth
        self.tp = Guillot2010(self.tp_guillot_T_irr,
                self.tp_guillot_kappa_ir,
                    self.tp_guillot_kappa_v1,self.tp_guillot_kappa_v2,self.tp_guillot_alpha)
        test_layers = 10

        pres_prof = np.ones(test_layers)
        



        self.tp.initialize_profile(Earth(),10,pres_prof)
         
 
    def test_compute_profile(self):
        self.tp.profile


if __name__ == '__main__':
    unittest.main()
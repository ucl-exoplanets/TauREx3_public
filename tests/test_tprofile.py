import unittest
import numpy as np
from taurex.data.profiles.temperature.tprofile import TemperatureProfile
from taurex.data.profiles.temperature import NPoint
from taurex.data.planet import Earth
 
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


class NpointTest(unittest.TestCase):


    
    def gen_npoint_test(self,num_points):
        import random

        temps = np.linspace(60,1000,num_points).tolist()
        pressure = np.linspace(40,100,num_points).tolist()
        NP = NPoint(temperature_points=temps,pressure_points=pressure)
        fitparams = NP.fitting_parameters()
        #print(fitparams)
        for idx,val in enumerate(zip(temps,pressure)) :
            T,P = val
            tpoint = 'T_point{}'.format(idx+1)
            ppoint = 'P_point{}'.format(idx+1)

            #Check the point is in the fitting param
            self.assertIn(tpoint,fitparams)
            self.assertIn(ppoint,fitparams)

            #Check we can get it
            self.assertEqual(NP[tpoint],temps[idx])
            self.assertEqual(NP[ppoint],pressure[idx])

            #Check we can set it
            NP[tpoint] = 400.0
            NP[ppoint] = 50.0

            #Check we can get it
            self.assertEqual(NP[tpoint],400.0)
            self.assertEqual(NP[ppoint],50.0)

            self.assertEqual(NP._t_points[idx],400.0)
            self.assertEqual(NP._p_points[idx],50.0)

        test_layers = 100

        pres_prof = np.ones(test_layers)

        NP.initialize_profile(Earth(),100,pres_prof)
        #See if this breaks
        NP.profile

    def test_exception(self):

        with self.assertRaises(Exception):
            NP = NPoint(temperature_points=[500.0,400.0],pressure_points=[100.0])

 
    def test_2layer(self):
        self.gen_npoint_test(1)

    def test_3layer(self):
        self.gen_npoint_test(2)

    def test_30layer(self):
        self.gen_npoint_test(29)



if __name__ == '__main__':
    unittest.main()
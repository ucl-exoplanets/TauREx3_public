import unittest
import numpy as np
from taurex.data.profiles.temperature.tprofile import TemperatureProfile
from taurex.data.profiles.temperature import NPoint
from taurex.data.planet import Earth
from taurex.data.profiles.temperature import Rodgers2000
from taurex.data.profiles.temperature.temparray import TemperatureArray
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


class RodgersTest(unittest.TestCase):

    def test_params(self):
        import random
        t_profile = np.arange(1,101,dtype=np.float)
        rp=Rodgers2000(temperature_layers=t_profile)
        params = rp.fitting_parameters()
        for idx,val in enumerate(t_profile):
            t_name = 'T_{}'.format(idx+1)
            self.assertIn(t_name,params)
            self.assertEqual(rp[t_name],val)

            rand_val = random.uniform(10,1000)

            rp[t_name] = rand_val
            self.assertEqual(rp[t_name],rand_val)
            self.assertEqual(rp._T_layers[idx],rand_val)


    def test_profile(self):
        t_profile = np.arange(1,101,dtype=np.float)

        rp=Rodgers2000(temperature_layers=t_profile)
        
        test_layers = 100
        pres_prof = np.ones(test_layers)

        rp.initialize_profile(Earth(),test_layers,pres_prof)

        rp.profile


class TemperatureArrayTest(unittest.TestCase):

    def test_basic(self):

        ag = TemperatureArray(tp_array=[200.0, 100.0])

        ag.initialize_profile(None, 2, None)

        self.assertEqual(ag.profile[0], 200)
        self.assertEqual(ag.profile[-1], 100)

    
    def test_interpolation(self):
        ag = TemperatureArray(tp_array=[200.0, 100.0])

        ag.initialize_profile(None, 3, None)

        self.assertEqual(ag.profile[0], 200)
        self.assertEqual(ag.profile[1], 150)
        self.assertEqual(ag.profile[-1], 100)


if __name__ == '__main__':
    unittest.main()
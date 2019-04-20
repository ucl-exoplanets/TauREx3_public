import unittest
import numpy as np
from taurex.data.profiles.gasprofiles.gasprofile import GasProfile,TaurexGasProfile


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


        #numpy.testing.ass

    def test_log_mode(self):
        with self.assertRaises(AttributeError):
            self.tp.setLinearLogMode('APPLES')
        
        self.tp.setLinearLogMode('linear')
        self.assertFalse(self.tp.isInLogMode)
        self.tp.setLinearLogMode('log')
        self.assertTrue(self.tp.isInLogMode)

        self.tp.setLinearLogMode('LINEAR')
        self.assertFalse(self.tp.isInLogMode)
        self.tp.setLinearLogMode('LOG')
        self.assertTrue(self.tp.isInLogMode)

        self.tp.setLinearLogMode('LiNeAR')
        self.assertFalse(self.tp.isInLogMode)
        self.tp.setLinearLogMode('LoG')
        self.assertTrue(self.tp.isInLogMode)

        #Test values are correct
        #Check linear mode first
        self.tp.setLinearLogMode('Linear')
        self.assertEqual(self.tp.readableValue(1e-4),1e-4)
        self.assertEqual(self.tp.writeableValue(1e-8),1e-8)

        #Now check log mode
        self.tp.setLinearLogMode('Log')
        self.assertEqual(self.tp.readableValue(1e-4),-4)
        self.assertEqual(self.tp.writeableValue(-8),1e-8)


class TaurexProfileTest(unittest.TestCase):

    def setUp(self):
 
        self.tp = TaurexGasProfile('test',['H2O','CH4'],[1e-4,1e-12])
    
        test_layers = 10

        pres_prof = np.ones(test_layers)

        self.tp.initialize_profile(10,pres_prof,pres_prof,pres_prof)

    def test_compute_active(self):
        
        self.tp.compute_active_gas_profile()
    
    def test_compute_inactive(self):
        self.tp.compute_inactive_gas_profile()
    


    def test_get_profiles(self):
        
        self.assertIsNotNone(self.tp.activeGasMixProfile)
        self.assertEqual(self.tp.activeGasMixProfile.shape[0],2)
        self.assertEqual(self.tp.activeGasMixProfile.shape[1],10)

        self.assertIsNotNone(self.tp.inActiveGasMixProfile)
        self.assertEqual(self.tp.inActiveGasMixProfile.shape[0],3)
        self.assertEqual(self.tp.inActiveGasMixProfile.shape[1],10)


        self.assertEqual(self.tp.muProfile.shape[0],10)
        zero_mu =np.zeros_like(self.tp.muProfile)
        self.assertFalse((zero_mu == self.tp.muProfile).all())

    def test_fitparams(self):
        params = self.tp.fitting_parameters()
        self.assertIn('N2',params)
        self.assertIn('H2_He',params)

        self.assertEqual(params['N2'][1],'N$_2$')



        self.tp.add_active_gas_param(0)
        self.tp.add_active_gas_param(1)
        params = self.tp.fitting_parameters()
        self.assertIn('H2O',params)
        self.assertIn('CH4',params)
        h2o = params['H2O']
        ch4 = params['CH4']
        self.assertEqual(h2o[2](),1e-4)
        self.assertEqual(ch4[2](),1e-12)
        h2o[3](1e-6)
        ch4[3](1e-18)
        self.assertEqual(self.tp.active_gas_mix_ratio[0],1e-6)
        self.assertEqual(self.tp.active_gas_mix_ratio[1],1e-18)

class ConstantProfileTest(unittest.TestCase):
    pass
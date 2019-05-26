import unittest
import numpy as np
from taurex.data.profiles.chemistry.gas.constantgas import ConstantGas
from taurex.data.profiles.chemistry.gas.twopointgas import TwoPointGas

class ConstantGasTest(unittest.TestCase):

    def test_init(self):
        gc = ConstantGas()
        self.assertEqual(gc.molecule,'H2O')
        self.assertEqual(gc.mixProfile,1e-5)
        param = gc.fitting_parameters()

        h2o=param['H2O']
        self.assertEqual(h2o[2](),1e-5)
        h2o[3](1e-4)
        self.assertEqual(h2o[2](),1e-4)
        self.assertEqual(gc.mixProfile,1e-4)

class TwoPointGasProfileTest(unittest.TestCase):

    def test_compute_profile(self):
        cgp = TwoPointGas('CH4',1e-4,1e-8)  
        params = cgp.fitting_parameters()



        test_layers = 10

        pres_prof = np.arange(1,test_layers+1)
        self.assertIsNone(cgp.mixProfile)
        cgp.initialize_profile(10,pres_prof,pres_prof,pres_prof)  
        self.assertIsNotNone(cgp.mixProfile)


    def test_fittingparams(self):
        cgp = TwoPointGas('CH4',1e-4,1e-8)  
        params = cgp.fitting_parameters()      

        self.assertIn('CH4_S',params)
        self.assertIn('CH4_T',params)
        self.assertEqual(params['CH4_S'][2](),1e-4)
        self.assertEqual(params['CH4_T'][2](),1e-8)
        params['CH4_S'][3](1e-5)
        params['CH4_T'][3](1e-7)
        self.assertEqual(params['CH4_S'][2](),1e-5)
        self.assertEqual(params['CH4_T'][2](),1e-7)       

        self.assertEqual(cgp.mixRatioSurface,1e-5)
        self.assertEqual(cgp.mixRatioTop,1e-7)     
# class TaurexProfileTest(unittest.TestCase):

#     def setUp(self):
 
#         self.tp = TaurexGasProfile('test',['H2O','CH4'],[1e-4,1e-12])
    
#         test_layers = 10

#         pres_prof = np.ones(test_layers)

#         self.tp.initialize_profile(10,pres_prof,pres_prof,pres_prof)

#     def test_compute_active(self):
        
#         self.tp.compute_active_gas_profile()
    
#     def test_compute_inactive(self):
#         self.tp.compute_inactive_gas_profile()
    


#     def test_get_profiles(self):
        
#         self.assertIsNotNone(self.tp.activeGasMixProfile)
#         self.assertEqual(self.tp.activeGasMixProfile.shape[0],2)
#         self.assertEqual(self.tp.activeGasMixProfile.shape[1],10)

#         self.assertIsNotNone(self.tp.inActiveGasMixProfile)
#         self.assertEqual(self.tp.inActiveGasMixProfile.shape[0],3)
#         self.assertEqual(self.tp.inActiveGasMixProfile.shape[1],10)


#         self.assertEqual(self.tp.muProfile.shape[0],10)
#         zero_mu =np.zeros_like(self.tp.muProfile)
#         self.assertFalse((zero_mu == self.tp.muProfile).all())

#     def test_fitparams(self):
#         params = self.tp.fitting_parameters()
#         self.assertIn('N2',params)
#         self.assertIn('H2_He',params)

#         self.assertEqual(params['N2'][1],'N$_2$')



#         self.tp.add_active_gas_param(0)
#         self.tp.add_active_gas_param(1)
#         params = self.tp.fitting_parameters()
#         self.assertIn('H2O',params)
#         self.assertIn('CH4',params)
#         h2o = params['H2O']
#         ch4 = params['CH4']
#         self.assertEqual(h2o[2](),1e-4)
#         self.assertEqual(ch4[2](),1e-12)
#         h2o[3](1e-6)
#         ch4[3](1e-18)
#         self.assertEqual(self.tp.active_gas_mix_ratio[0],1e-6)
#         self.assertEqual(self.tp.active_gas_mix_ratio[1],1e-18)

# class ConstantProfileTest(unittest.TestCase):
    


    
#     def test_init(self):
#         from taurex.data.profiles.gas import ConstantGasProfile
#         cgp = ConstantGasProfile(['H2O','CH4'],[1e-4,1e-12])
#         print('Constant---------',[x[2] for x in cgp.fitting_parameters().values()])
#         test_layers = 10

#         pres_prof = np.ones(test_layers)

#         cgp.initialize_profile(10,pres_prof,pres_prof,pres_prof)
    

#     def test_default_constant(self):
#         from taurex.data.profiles.gas import ConstantGasProfile
#         cgp = ConstantGasProfile()     

#         test_layers = 10

#         pres_prof = np.ones(test_layers)

#         cgp.initialize_profile(10,pres_prof,pres_prof,pres_prof)



# class ComplexProfileTest(unittest.TestCase):

        
           


#     def test_parameters(self):
#         cgp = ComplexGasProfile('test',['H2O','CH4'],[1e-4,1e-12],['CH4'],[1e-4],[1e-8])  
#         params = cgp.fitting_parameters()
#         self.assertIn('H2O',params)
#         self.assertNotIn('CH4',params)

#         self.assertIn('T CH4',params)
#         self.assertIn('S CH4',params)
#     # def test_log_parameters(self):
#     #     cgp = ComplexGasProfile('test',['H2O','CH4'],[1e-4,1e-12],['CH4'],[1e-4],[1e-8],mode='log')  
#     #     params = cgp.fitting_parameters()      

#     #     self.assertIn('log_H2O',params)
#     #     self.assertNotIn('log_CH4',params)

#     #     self.assertNotIn('T CH4',params)
#     #     self.assertNotIn('S CH4',params)
#     #     self.assertIn('T_log_CH4',params)
#     #     self.assertIn('S_log_CH4',params)
# class TwoPointGasProfileTest(unittest.TestCase):

#     def test_compute_profile(self):
#         cgp = TwoPointGasProfile(['H2O','CH4'],[1e-4,1e-12],['CH4'],[1e-4],[1e-8])  
#         params = cgp.fitting_parameters()
#         test_layers = 10

#         pres_prof = np.ones(test_layers)

#         cgp.initialize_profile(10,pres_prof,pres_prof,pres_prof)   

# class TwoLayerGasProfileTest(unittest.TestCase):

#     def test_compute_profile(self):
#         cgp = TwoLayerGasProfile(['H2O','CH4'],[1e-4,1e-12],['CH4'],[1e-4],[1e-8],[1e-4])  
#         params = cgp.fitting_parameters()


#         test_layers = 100

#         pres_prof = np.ones(test_layers)
#         self.assertIn('P CH4',params)
#         self.assertNotIn('P H2O',params)
#         cgp.initialize_profile(100,pres_prof,pres_prof,pres_prof)   


# class AceGasProfileTest(unittest.TestCase):

#     def test_compute_profile(self):
#         from taurex.data.profiles.gas.acegasprofile import ACEGasProfile
#         cgp = ACEGasProfile(['H2O','CH4'],spec_file='src/ACE/Data/composes.dat',therm_file='src/ACE/Data/NASA.therm')  
#         params = cgp.fitting_parameters()
#         test_layers = 10

#         pres_prof = np.ones(test_layers)

#         cgp.initialize_profile(10,pres_prof,pres_prof,pres_prof)   
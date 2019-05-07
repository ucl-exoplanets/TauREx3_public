import unittest
from taurex.parameter.factory import create_gas_profile

class GasFactoryTest(unittest.TestCase):

    def test_constant(self):
        from taurex.data.profiles.gas import ConstantGasProfile
        
        test_dict = {'profile_type': 'constant',
                    'mode': 'log'
                    
                        }
        
        gas = create_gas_profile(test_dict)
        self.assertIs(type(gas),ConstantGasProfile)
        self.assertTrue(gas.isInLogMode)

        test_dict = {'profile_type': 'constant',
                    'mode': 'linear',
                    'active_gases': ['H2O','CO2'],
                    'active_gas_mix_ratio' : [1e-4,1e-5],
                    'CO2' : 1e-7
                        }
        
        gas = create_gas_profile(test_dict)
        self.assertFalse(gas.isInLogMode)
        self.assertIn('CO2',gas.active_gases)    
        self.assertEqual(gas.active_gas_mix_ratio[1],1e-7)    
        self.assertNotEqual(gas.active_gas_mix_ratio[1],1e-5)

        test_dict = {'profile_type': 'constant',
                    'mode': 'log',
                    'active_gases': ['H2O','CO2'],
                    'active_gas_mix_ratio' : [1e-4,1e-5],
                    'log_H2O' : -7
                        }

        gas = create_gas_profile(test_dict)
        self.assertTrue(gas.isInLogMode)
        self.assertIn('CO2',gas.active_gases)    
        self.assertEqual(gas.active_gas_mix_ratio[1],1e-5)    
        self.assertEqual(gas.active_gas_mix_ratio[0],1e-7)
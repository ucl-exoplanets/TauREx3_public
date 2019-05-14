import unittest
from taurex.parameter.factory import create_gas_profile

class GasFactoryTest(unittest.TestCase):

    def test_constant(self):
        from taurex.data.profiles.gas import ConstantGasProfile
        
        test_dict = {'profile_type': 'constant',
                    
                        }
        
        gas = create_gas_profile(test_dict)
        self.assertIs(type(gas),ConstantGasProfile)

        test_dict = {'profile_type': 'constant',
                    'mode': 'linear',
                    'active_gases': ['H2O','CO2'],
                    'active_gas_mix_ratio' : [1e-4,1e-5],
                    'CO2' : 1e-7
                        }
        
        gas = create_gas_profile(test_dict)
        self.assertIn('CO2',gas.active_gases)    
        self.assertEqual(gas.active_gas_mix_ratio[1],1e-7)    
        self.assertNotEqual(gas.active_gas_mix_ratio[1],1e-5)

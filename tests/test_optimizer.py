import unittest
from taurex.optimizer.optimizer import Optimizer
from taurex.model import TransmissionModel
from unittest.mock import patch


class OptimizerTest(unittest.TestCase):


    def test_fit_params(self):
        from taurex.cache import OpacityCache
        #spin up a model
        with patch.object(OpacityCache,"find_list_of_molecules") as mock_my_method:
            mock_my_method.return_value = ['H2O','CH4']
            tm = TransmissionModel()
            tm.build()
        opt = Optimizer('test',model=tm)

        
        opt.enable_fit('H2O')
        opt.enable_fit('T')  

        opt.compile_params()    
        names = opt.fit_names
        self.assertIn('T',names)
        self.assertIn('log_H2O',names)

        tm['T'] = 2000.0
        tm['H2O'] = 0.01

        values = opt.fit_values
        opt.fit_boundaries
        opt.fit_latex
        opt.fit_values_nomode

        opt.set_mode('H2O','linear')
        opt.set_mode('T','log')  
        opt.compile_params()    
        
        names = opt.fit_names
        self.assertIn('log_T',names)
        self.assertIn('H2O',names)  

        opt.set_mode('T','linear')
        opt.set_mode('H2O','log')
        opt.set_factor_boundary('H2O',(0.5,1.5))
        opt.set_factor_boundary('T',(0.5,1.5))
        opt.compile_params()  
        h2o_index = opt.fit_names.index('log_H2O')
        t_index = opt.fit_names.index('T')
        
    

        self.assertEqual(opt.fit_boundaries[t_index][0],1000.0)
        self.assertEqual(opt.fit_boundaries[t_index][1],3000.0)
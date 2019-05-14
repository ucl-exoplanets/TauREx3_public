import unittest
from taurex.optimizer.optimizer import Optimizer
from taurex.model import TransmissionModel



class OptimizerTest(unittest.TestCase):


    def test_fit_params(self):

        #spin up a model
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
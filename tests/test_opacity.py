import unittest
from unittest.mock import patch, mock_open
from taurex.opacity.opacity import Opacity
from taurex.opacity.pickleopacity import PickleOpacity
import numpy as np
import logging


pickle_test_data ={'t': np.arange(1,26),
     'p': np.arange(1,26),
     'name': 'testMol',
      'wno': np.linspace(0,10000,1000),
      'xsecarr': np.random.rand(25,25,1000)}

class OpacityTest(unittest.TestCase):


    def test_notimplemented(self):
        op = Opacity('My name')
        with self.assertRaises(NotImplementedError):
            op.opacity(None,100,200)




class PickleOpacityTest(unittest.TestCase):
    

    def setUp(self):
        import pickle
        data = pickle.dumps(pickle_test_data)
        self.pop = None
        with patch("builtins.open", mock_open(read_data=data)) as mock_file:
            self.pop=PickleOpacity('unittestfile')
    
    def test_properties(self):

        
        self.assertEqual(self.pop.moleculeName,'TESTMOL')
        np.testing.assert_equal(pickle_test_data['t'],self.pop.temperatureGrid)
        np.testing.assert_equal(pickle_test_data['p']*1e5,self.pop.pressureGrid)
        np.testing.assert_equal(pickle_test_data['wno'],self.pop.wavenumberGrid)
        np.testing.assert_equal(pickle_test_data['xsecarr'],self.pop._xsec_grid)


    def test_opacity_calc(self):
        logging.basicConfig(level=logging.DEBUG)
        t_list = pickle_test_data['t']
        p_list = pickle_test_data['p']

        for idx,vals in enumerate(zip(t_list,p_list)):
            t_idx,p_idx = vals
            xsec = pickle_test_data['xsecarr'][idx,idx]
            np.testing.assert_almost_equal(xsec/10000,self.pop.compute_opacity(t_idx,p_idx*1e5))


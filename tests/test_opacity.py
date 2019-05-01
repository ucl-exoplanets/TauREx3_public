import unittest
from unittest.mock import patch, mock_open
from taurex.opacity.opacity import Opacity
from taurex.opacity.pickleopacity import PickleOpacity
import numpy as np

pickle_test_data ={'t': np.arange(0,20),
     'p': np.arange(0,25),
     'name': 'testMol',
      'wno': np.linspace(0,10000,1000),
      'xsecarr': np.random.rand(25,20,1000)}

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

        
        self.assertEqual(self.pop.moleculeName,'testMol')
        np.testing.assert_equal(pickle_test_data['t'],self.pop.temperatureGrid)
        np.testing.assert_equal(pickle_test_data['p'],self.pop.pressureGrid)
        np.testing.assert_equal(pickle_test_data['wno'],self.pop.wavenumberGrid)
        np.testing.assert_equal(pickle_test_data['xsecarr'],self.pop._xsec_grid)


    def test_opacity_calc(self):

        t_list = [0,4,7]
        p_list = [3,6,9]

        for t_idx,p_idx in zip(t_list,p_list):
            xsec = pickle_test_data['xsecarr'][p_idx,t_idx]
            np.testing.assert_equal(xsec,self.pop.compute_opacity(t_idx,p_idx))


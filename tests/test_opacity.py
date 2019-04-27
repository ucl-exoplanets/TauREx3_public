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
            op.opacity(100,200)




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


        
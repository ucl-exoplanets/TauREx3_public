
import unittest
import shutil, tempfile
from os import path
from unittest.mock import patch, mock_open
from taurex.model.model import ForwardModel,SimpleForwardModel
import numpy as np
import pickle

class ForwardModelTest(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.gen_opacities(10)

    

    def gen_opacities(self,num_opacties):
        from taurex.opacity import PickleOpacity
        self.opacity_names = ['op_test{}'.format(x) for x in range(num_opacties)]
        self.opacity_list = []
        for op_name in self.opacity_names:
            pickle_test_data ={'t': np.arange(0,20),
                'p': np.arange(0,25),
                'name': op_name,
                'wno': np.linspace(0,10000,1000),
                'xsecarr': np.random.rand(25,20,1000)}
            with open(path.join(self.test_dir, '{}.pickle'.format(op_name)), 'wb') as f:
                pickle.dump(pickle_test_data,f)
            self.opacity_list.append(PickleOpacity(path.join(self.test_dir, '{}.pickle'.format(op_name))))

    




        
    def test_init(self):
        model = ForwardModel('test')
    
    def test_load_opacities(self):


        #Test single load
        model = ForwardModel('test')
        model.load_opacities(self.opacity_list[0],None)
        self.assertIn('op_test0',model.opacity_dict)

        #Test list_load
        model = ForwardModel('test')
        model.load_opacities(self.opacity_list,None)
        for op_name in self.opacity_names:
            self.assertIn(op_name,model.opacity_dict)    

        #Test path_load
        model = ForwardModel('test')
        model.load_opacities(None,self.test_dir)
        for op_name in self.opacity_names:
            self.assertIn(op_name,model.opacity_dict)    

    def tearDown(self):
        shutil.rmtree(self.test_dir)



class SimpleForwardModelTest(unittest.TestCase):


    def test_init(self):
        model = SimpleForwardModel('test')
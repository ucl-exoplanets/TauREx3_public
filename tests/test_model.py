
import unittest
import shutil, tempfile
from os import path
from unittest.mock import patch, mock_open
from taurex.model.model import ForwardModel
from taurex.model.simplemodel import SimpleForwardModel
import numpy as np
import pickle

class ForwardModelTest(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.gen_opacities(10)
        self.gen_cia(10)
    

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


    def gen_cia(self,num_cia):
        from taurex.cia import PickleCIA
        self.cia_names = ['cia_test{}'.format(x) for x in range(num_cia)]
        self.cia_list = []
        for op_name in self.cia_names:
            pickle_test_data ={'t': np.arange(0,20),
                'wno': np.linspace(0,10000,1000),
                'xsecarr': np.random.rand(20,1000)}
            with open(path.join(self.test_dir, '{}.db'.format(op_name)), 'wb') as f:
                pickle.dump(pickle_test_data,f)
            self.cia_list.append(PickleCIA(path.join(self.test_dir, '{}.db'.format(op_name)),op_name))





        
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
        
        self.assertNotIn('cia_test0',model.opacity_dict)

        with self.assertRaises(Exception):
            model.add_opacity(self.opacity_list[0])

        model = ForwardModel('test')
        model.load_opacities(None,self.test_dir,molecule_filter=['op_test0','op_test2'])
        self.assertIn('op_test0',model.opacity_dict)    
        self.assertIn('op_test2',model.opacity_dict)    
        self.assertNotIn('op_test1',model.opacity_dict)    
    def test_load_cia(self):


        #Test single load
        model = ForwardModel('test')
        model.load_cia(self.cia_list[0],None)
        self.assertIn('cia_test0',model.cia_dict)

        #Test list_load
        model = ForwardModel('test')
        model.load_cia(self.cia_list,None)
        for op_name in self.cia_names:
            self.assertIn(op_name,model.cia_dict)    

        #Test path_load
        model = ForwardModel('test')
        model.load_cia(None,self.test_dir)
        for op_name in self.cia_names:
            self.assertIn(op_name,model.cia_dict)    
        
        self.assertNotIn('op_test0',model.cia_dict)

        with self.assertRaises(Exception):
            model.add_cia(self.cia_list[0])


    def tearDown(self):
        shutil.rmtree(self.test_dir)



class SimpleForwardModelTest(unittest.TestCase):


    def test_init(self):
        model = SimpleForwardModel('test')
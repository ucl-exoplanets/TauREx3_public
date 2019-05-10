
import unittest
import shutil, tempfile
from os import path
import pickle
from unittest.mock import patch, mock_open
from taurex.data.stellar.phoenix import PhoenixStar
import numpy as np
import logging

class TestPhoenixStar(unittest.TestCase):


    def setUp(self):

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self._t_list = [100.0,4999.0,400.0,6000.0,3540.0,5500.0,8000.0]
        self.gen_temps(self._t_list )
        self._t_list.sort()
        #self.gen_cia(10)
    

    def gen_temps(self,temperatures):
        self.temps_names = [path.join(self.test_dir,'lte{}-jsadhfaksjdf.fmt'.format(int(x))) for x in temperatures]
        self.temps_list = []
        for t_name in self.temps_names:
            arr = np.random.rand(1000,3)
            np.savetxt(t_name,arr)


    
    def test_sort_temps(self):
        ps = PhoenixStar(phoenix_path=self.test_dir)
        
        list_out = ps.detect_all_T(self.test_dir)
        new_list = [x[0] for x in list_out]
        self.assertEqual(set(self._t_list),set(new_list))




    def tearDown(self):
        shutil.rmtree(self.test_dir)

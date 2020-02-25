
import unittest
import shutil
import tempfile
from os import path
import pickle
from unittest.mock import patch, mock_open
from taurex.cache.opacitycache import OpacityCache
from taurex.cache.ciaacache import CIACache
from taurex.cache.globalcache import GlobalCache
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


class TestOpacityCache(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.gen_opacities(10)
        # self.gen_cia(10)

    def gen_opacities(self, num_opacties):
        from taurex.opacity import PickleOpacity
        self.opacity_names = ['optest{}'.format(
            x) for x in range(num_opacties)]
        self.opacity_list = []
        for op_name in self.opacity_names:
            pickle_test_data = {'t': np.arange(0, 20),
                                'p': np.arange(0, 25),
                                'name': op_name,
                                'wno': np.linspace(0, 10000, 1000),
                                'xsecarr': np.random.rand(25, 20, 1000)}
            with open(path.join(self.test_dir, '{}.pickle'.format(op_name)), 'wb') as f:
                pickle.dump(pickle_test_data, f)
            self.opacity_list.append(PickleOpacity(
                path.join(self.test_dir, '{}.pickle'.format(op_name))))

    # def gen_cia(self,num_cia):
    #     from taurex.cia import PickleCIA
    #     self.cia_names = ['cia_test{}'.format(x) for x in range(num_cia)]
    #     self.cia_list = []
    #     for op_name in self.cia_names:
    #         pickle_test_data ={'t': np.arange(0,20),
    #             'wno': np.linspace(0,10000,1000),
    #             'xsecarr': np.random.rand(20,1000)}
    #         with open(path.join(self.test_dir, '{}.db'.format(op_name)), 'wb') as f:
    #             pickle.dump(pickle_test_data,f)
    #         self.cia_list.append(PickleCIA(path.join(self.test_dir, '{}.db'.format(op_name)),op_name))

    def test_load_opacities(self):
        opacity1 = OpacityCache()
        opacity2 = OpacityCache()

        self.assertEqual(opacity1, opacity2)

        opacity1.set_opacity_path(self.test_dir)

        self.assertEqual(opacity1._opacity_path, opacity2._opacity_path)

        opacity1['optest0']

        self.assertIn('optest0', opacity2.opacity_dict)

        opacity2['optest2']
        self.assertIn('optest2', opacity1.opacity_dict)

    def test_find_molecules(self):
        opacity1 = OpacityCache()
        opacity1.set_opacity_path(self.test_dir)

        opList = opacity1.find_list_of_molecules()

        self.assertIn('optest2', opList)

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class TestCIACache(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        # self.gen_opacities(10)
        self.gen_cia(10)

    # def gen_opacities(self,num_opacties):
    #     from taurex.opacity import PickleOpacity
    #     self.opacity_names = ['optest{}'.format(x).upper() for x in range(num_opacties)]
    #     self.opacity_list = []
    #     for op_name in self.opacity_names:
    #         pickle_test_data ={'t': np.arange(0,20),
    #             'p': np.arange(0,25),
    #             'name': op_name,
    #             'wno': np.linspace(0,10000,1000),
    #             'xsecarr': np.random.rand(25,20,1000)}
    #         with open(path.join(self.test_dir, '{}.pickle'.format(op_name)), 'wb') as f:
    #             pickle.dump(pickle_test_data,f)
    #         self.opacity_list.append(PickleOpacity(path.join(self.test_dir, '{}.pickle'.format(op_name))))

    def gen_cia(self, num_cia):
        from taurex.cia import PickleCIA
        self.cia_names = ['cia{}_test'.format(x) for x in range(num_cia)]
        self.cia_list = []
        for op_name in self.cia_names:
            pickle_test_data = {'t': np.arange(0, 20),
                                'wno': np.linspace(0, 10000, 1000),
                                'xsecarr': np.random.rand(20, 1000)}
            with open(path.join(self.test_dir, '{}.db'.format(op_name)), 'wb') as f:
                pickle.dump(pickle_test_data, f)
            self.cia_list.append(
                PickleCIA(path.join(self.test_dir, '{}.db'.format(op_name)), op_name))

    def test_load_cia(self):
        cia1 = CIACache()
        cia2 = CIACache()

        self.assertEqual(cia1, cia2)

        cia1.set_cia_path(self.test_dir)

        self.assertEqual(cia1._cia_path, cia2._cia_path)

        cia1['cia0']

        self.assertIn('cia0', cia2.cia_dict)

        cia2['cia2']
        self.assertIn('cia2', cia1.cia_dict)

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class TestGlobalCache(unittest.TestCase):

    def test_read_write(self):

        gc = GlobalCache()

        gc['TESTVARIABLE'] = 100

        self.assertEqual(gc['TESTVARIABLE'], 100)

        self.assertEqual(GlobalCache()['TESTVARIABLE'], 100)

        self.assertIsNone(GlobalCache()['ANOTHERVARIABLE'])

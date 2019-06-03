import unittest
from unittest.mock import patch, mock_open
from taurex.data.spectrum.lightcurve import ObservedLightCurve
import numpy as np
import logging



class ObsLightCurveTest(unittest.TestCase):


    def create_instrument(self,num_elems,instruments):
        import pickle

        lc_dict = {}
        lc_dict['lc_info'] = np.random.rand(num_elems,10)
        lc_dict['data']={}
        for ins in instruments:
            lc_dict['data'][ins] = np.random.rand(num_elems*2,83)
        
        return lc_dict,pickle.dumps(lc_dict)

    

    def test_init(self):
        
        data,pickled = self.create_instrument(40,['wfc3'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')
        


    def test_instruments(self):
        
        data,pickled = self.create_instrument(40,['wfc3'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')
        
        self.assertEqual(obs.spectrum.shape[0],40)
        self.assertEqual(obs.spectrum.shape[1],83)

        self.assertEqual(obs.errorBar.shape[0],40)
        self.assertEqual(obs.errorBar.shape[1],83)


        data,pickled = self.create_instrument(40,['wfc3','spitzer'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')        

        self.assertEqual(obs.spectrum.shape[0],80)
        self.assertEqual(obs.spectrum.shape[1],83)

        self.assertEqual(obs.errorBar.shape[0],80)
        self.assertEqual(obs.errorBar.shape[1],83)

        data,pickled = self.create_instrument(40,['wfc3','spitzer','stis'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')        

        self.assertEqual(obs.spectrum.shape[0],120)
        self.assertEqual(obs.spectrum.shape[1],83)

        self.assertEqual(obs.errorBar.shape[0],120)
        self.assertEqual(obs.errorBar.shape[1],83)


        data,pickled = self.create_instrument(40,['wfc3','spitzer','stis','unknown'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')        

        self.assertEqual(obs.spectrum.shape[0],120)
        self.assertEqual(obs.spectrum.shape[1],83)

        self.assertEqual(obs.errorBar.shape[0],120)
        self.assertEqual(obs.errorBar.shape[1],83)

    
    def test_ordering(self):
        
        data,pickled = self.create_instrument(40,['wfc3'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['data']['wfc3'][:40],obs.spectrum)

        data,pickled = self.create_instrument(40,['wfc3','spitzer'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['data']['wfc3'][:40],obs.spectrum[:40])
        np.testing.assert_equal(data['data']['spitzer'][:40],obs.spectrum[40:])



        data,pickled = self.create_instrument(40,['wfc3','spitzer','stis'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['data']['wfc3'][:40],obs.spectrum[:40])
        np.testing.assert_equal(data['data']['spitzer'][:40],obs.spectrum[40:80])
        np.testing.assert_equal(data['data']['stis'][:40],obs.spectrum[80:])


        data,pickled = self.create_instrument(40,['spitzer','stis'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['data']['spitzer'][:40],obs.spectrum[0:40])
        np.testing.assert_equal(data['data']['stis'][:40],obs.spectrum[40:])

        data,pickled = self.create_instrument(40,['wfc3','stis'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['data']['wfc3'][:40],obs.spectrum[0:40])
        np.testing.assert_equal(data['data']['stis'][:40],obs.spectrum[40:])
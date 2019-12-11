import unittest
from unittest.mock import patch, mock_open
from taurex.data.spectrum.lightcurve import ObservedLightCurve
import numpy as np
import logging



class ObsLightCurveTest(unittest.TestCase):


    def create_instrument(self,num_elems,instruments):
        import pickle

        inst_dict = {
            'wfc3': np.linspace(1.1, 1.8, num_elems),
            'spitzer': np.linspace(3.4, 8.2, num_elems),
            'stis': np.linspace(0.3, 1.0, num_elems),
            'twinkle': np.linspace(0.4, 4.5, num_elems),
        }

        lc_dict = {}
        lc_dict['obs_spectrum'] = np.random.rand(num_elems,10)
        for ins in instruments:
            lc_dict[ins] = {}
            lc_dict[ins]['data'] = np.random.rand(num_elems,num_elems,4)
            lc_dict[ins]['wl_grid'] = inst_dict.get(ins,np.random.rand(num_elems))
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

        self.assertEqual(obs.errorBar.shape[0],40)


        data,pickled = self.create_instrument(40,['wfc3','spitzer'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')        

        self.assertEqual(obs.spectrum.shape[0],80)

        data,pickled = self.create_instrument(40,['wfc3','spitzer','stis'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')        

        self.assertEqual(obs.spectrum.shape[0],120)


        data,pickled = self.create_instrument(40,['wfc3','spitzer','stis','unknown'])

        
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')        

        self.assertEqual(obs.spectrum.shape[0],120)

    
    def test_ordering(self):
        
        data,pickled = self.create_instrument(40,['wfc3'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['wfc3']['data'][:40,:,0],obs.spectrum)

        data,pickled = self.create_instrument(40,['wfc3','spitzer'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['wfc3']['data'][:40,:,0],obs.spectrum[40:])
        np.testing.assert_equal(data['spitzer']['data'][:40,:,0],obs.spectrum[:40])



        data,pickled = self.create_instrument(40,['wfc3','spitzer','stis'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     



        np.testing.assert_equal(data['stis']['data'][:40,:,0],obs.spectrum[80:])
        np.testing.assert_equal(data['spitzer']['data'][:40,:,0],obs.spectrum[:40])
        np.testing.assert_equal(data['wfc3']['data'][:40,:,0],obs.spectrum[40:80])

        data,pickled = self.create_instrument(40,['spitzer','stis'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['spitzer']['data'][:40,:,0],obs.spectrum[0:40])
        np.testing.assert_equal(data['stis']['data'][:40,:,0],obs.spectrum[40:])

        data,pickled = self.create_instrument(40,['wfc3','stis'])
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            obs = ObservedLightCurve('testdata')     

        np.testing.assert_equal(data['wfc3']['data'][:40,:,0],obs.spectrum[0:40])
        np.testing.assert_equal(data['stis']['data'][:40,:,0],obs.spectrum[40:])
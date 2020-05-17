import unittest
from unittest.mock import patch, mock_open
from unittest import mock
from taurex.model.lightcurve.lightcurve import LightCurveData, LightCurveModel
from taurex.model import TransmissionModel
import numpy as np
import logging


class LightCurveDataTest(unittest.TestCase):

    def create_instrument(self, instruments, num_elems, num_timeseries):
        import pickle

        inst_dict = {
            'wfc3': np.linspace(1.1, 1.8, num_elems),
            'spitzer': np.linspace(3.4, 8.2, num_elems),
            'stis': np.linspace(0.3, 1.0, num_elems),
            'twinkle': np.linspace(0.4, 4.5, num_elems),
        }

        lc_dict = {}
        lc_dict['obs_spectrum'] = np.random.rand(num_elems, 10)
        for ins in instruments:
            lc_dict[ins] = {}
            lc_dict[ins]['time_series'] = np.random.rand(num_timeseries)
            lc_dict[ins]['data'] = np.random.rand(num_elems, num_elems, 4)
            lc_dict[ins]['wl_grid'] = inst_dict.get(
                ins, np.random.rand(num_elems))
        return lc_dict, pickle.dumps(lc_dict)

    def test_init(self):

        data, pickled = self.create_instrument(['wfc3'], 25, 83)

        obs = LightCurveData(data, 'wfc3', (1.1, 1.5))

        with self.assertRaises(KeyError):
            obs = LightCurveData(data, 'WFC2', (1.1, 1.5))

    def test_factory(self):

        data, pickled = self.create_instrument(['wfc3', 'spitzer'], 25, 83)

        obs = LightCurveData.fromInstrumentName('wfc3', data)
        self.assertEqual(obs.instrumentName, 'wfc3')

        min_wn, max_wn = obs.wavelengthRegion
        self.assertEqual(min_wn, 1.1)
        self.assertEqual(max_wn, 1.8)

        obs = LightCurveData.fromInstrumentName('spitzer', data)

        min_wn, max_wn = obs.wavelengthRegion
        self.assertEqual(min_wn, 3.4)
        self.assertEqual(max_wn, 8.2)

        self.assertEqual(obs.instrumentName, 'spitzer')
        obs = LightCurveData.fromInstrumentName('SPITZER', data)
        self.assertEqual(obs.instrumentName, 'spitzer')
        obs = LightCurveData.fromInstrumentName('WFC3', data)
        self.assertEqual(obs.instrumentName, 'wfc3')

        with self.assertRaises(KeyError):
            obs = LightCurveData.fromInstrumentName('WFC2', data)


class LightCurveTest(unittest.TestCase):

    def create_lc_pickle(self, instruments, num_elems, num_timeseries):
        import pickle

        inst_dict = {
            'wfc3': np.linspace(1.1, 1.8, num_elems),
            'spitzer': np.linspace(3.4, 8.2, num_elems),
            'stis': np.linspace(0.3, 1.0, num_elems),
            'twinkle': np.linspace(0.4, 4.5, num_elems),
        }

        lc_dict = {}
        lc_dict['obs_spectrum'] = np.random.rand(num_elems, 10)
        for ins in instruments:
            lc_dict[ins] = {}
            lc_dict[ins]['time_series'] = np.random.rand(num_timeseries)
            lc_dict[ins]['data'] = np.random.rand(num_elems, num_elems, 4)
            lc_dict[ins]['wl_grid'] = inst_dict.get(
                ins, np.random.rand(num_elems))
            lc_dict[ins]['ld_coeff'] = np.random.rand(num_elems, 4)
        lc_dict['orbital_info'] = {}
        lc_dict['orbital_info']['mt'] = 1.0
        lc_dict['orbital_info']['i'] = 1.0
        lc_dict['orbital_info']['period'] = 1.0
        lc_dict['orbital_info']['periastron'] = 1.0
        lc_dict['orbital_info']['sma_over_rs'] = 1.0
        lc_dict['orbital_info']['e'] = 1.0
        return lc_dict, pickle.dumps(lc_dict)

    def create_lightcurve(self, pickled):
        tm = TransmissionModel()
        with patch("builtins.open", mock_open(read_data=pickled)) as mock_file:
            lc = LightCurveModel(tm, 'HAHAFUNNY')
        return lc

    def test_init(self):

        data, pickled = self.create_lc_pickle(['wfc3'], 25, 83)

        lc = self.create_lightcurve(pickled)

        data, pickled = self.create_lc_pickle(
            ['wfc3', 'spitzer', 'stis'], 25, 83)

        lc = self.create_lightcurve(pickled)

    def test_params(self):
        data, pickled = self.create_lc_pickle(['wfc3'], 25, 83)

        lc = self.create_lightcurve(pickled)
        lc.build()
        params = lc.fittingParameters

        for x in range(25):
            nfactor = 'Nfactor_{}'.format(x)

            self.assertIn(nfactor, params)

        data, pickled = self.create_lc_pickle(['wfc3', 'spitzer'], 25, 83)

        lc = self.create_lightcurve(pickled)
        lc.build()
        params = lc.fittingParameters

        for x in range(50):
            nfactor = 'Nfactor_{}'.format(x)

            self.assertIn(nfactor, params)
        params = lc.fittingParameters
        self.assertIn('H2O', params)
        self.assertIn('CH4', params)
        self.assertIn('T', params)

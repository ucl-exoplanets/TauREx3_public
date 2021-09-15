import pytest
import unittest
from taurex.data.profiles.temperature import NPoint
from taurex.data.planet import Earth
from ..strategies import TP_npoints
from hypothesis import given, note, strategies as st
from taurex.exceptions import InvalidModelException
import numpy as np


@given(params=TP_npoints(), limit_slope=st.floats(10.0, 999999.0, allow_nan=False),
       smoothing_window=st.integers(1, 100))
def test_npoint(params, limit_slope, smoothing_window):

    nlayers, T_top, T_surface, P_top, P_surface, \
        temp_points, press_points, P = params

    planet = Earth()

    npoint = NPoint(T_surface=T_surface, T_top=T_top,
                    P_surface=P_surface, P_top=P_top,
                    temperature_points=temp_points,
                    pressure_points=press_points,
                    limit_slope=limit_slope,
                    smoothing_window=smoothing_window)

    # Test params
    npoints = len(temp_points)

    params = npoint.fitting_parameters()

    for x in range(npoints):
        assert f'T_point{x+1}' in params
        assert params[f'T_point{x+1}'][2]() == temp_points[x]
        assert f'P_point{x+1}' in params
        assert params[f'P_point{x+1}'][2]() == press_points[x]

    npoint.initialize_profile(planet=planet, nlayers=nlayers,
                              pressure_profile=P)

    Pnodes = [P[0], *press_points, P_top]
    Tnodes = [T_surface, *temp_points, T_top]
    diff = np.diff(Tnodes)/np.diff(np.log10(Pnodes))
    if any(Pnodes[i] <= Pnodes[i+1]
            for i in range(len(Pnodes)-1)):

        with pytest.raises(InvalidModelException):
            npoint.profile
        
    elif any(np.abs(diff) >= limit_slope):
        with pytest.raises(InvalidModelException):
            npoint.profile
    else:
        # Lets make sure it doesn't crash
        npoint.profile




# class NpointTest(unittest.TestCase):

#     def gen_npoint_test(self, num_points):
#         import random

#         temps = np.linspace(60, 1000, num_points).tolist()
#         pressure = np.linspace(40, 10, num_points).tolist()

#         NP = NPoint(temperature_points=temps, pressure_points=pressure)
#         fitparams = NP.fitting_parameters()

#         test_layers = 100

#         pres_prof = np.logspace(6, 0, 100)

#         NP.initialize_profile(Earth(), 100, pres_prof)
#         # See if this breaks
#         NP.profile
    
#         # print(fitparams)
#         for idx, val in enumerate(zip(temps, pressure)):
#             T, P = val
#             tpoint = 'T_point{}'.format(idx+1)
#             ppoint = 'P_point{}'.format(idx+1)

#             # Check the point is in the fitting param
#             self.assertIn(tpoint, fitparams)
#             self.assertIn(ppoint, fitparams)

#             # Check we can get it
#             self.assertEqual(NP[tpoint], temps[idx])
#             self.assertEqual(NP[ppoint], pressure[idx])

#             # Check we can set it
#             NP[tpoint] = 400.0
#             NP[ppoint] = 50.0

#             # Check we can get it
#             self.assertEqual(NP[tpoint], 400.0)
#             self.assertEqual(NP[ppoint], 50.0)

#             self.assertEqual(NP._t_points[idx], 400.0)
#             self.assertEqual(NP._p_points[idx], 50.0)



#     def test_exception(self):

#         with self.assertRaises(Exception):
#             NP = NPoint(temperature_points=[
#                         500.0, 400.0], pressure_points=[100.0])

#     def test_invalid_exception(self):
#         from taurex.data.profiles.temperature.npoint import InvalidTemperatureException

#         with self.assertRaises(InvalidTemperatureException):

#             NP = NPoint(T_surface=1000, T_top=1000, P_top=1e1, pressure_points=[1e3,1e4],
#                         temperature_points=[1000,1000])

#             pressure_profile = np.logspace(6, 0, 100)

#             NP.initialize_profile(pressure_profile=pressure_profile, nlayers=100)

#             NP.profile



#         NP = NPoint(T_surface=1000, T_top=1000, P_top=1e1, pressure_points=[1e4,1e3],
#                     temperature_points=[1000,1000], limit_slope=1000)

#         pressure_profile = np.logspace(6, 0, 100)

#         NP.initialize_profile(pressure_profile=pressure_profile, nlayers=100)

#         NP.profile
#         with self.assertRaises(InvalidTemperatureException):
#             NP['T_point1'] = 100000

#             NP.profile
        

#     def test_2layer(self):
#         self.gen_npoint_test(1)

#     def test_3layer(self):
#         self.gen_npoint_test(2)

#     def test_30layer(self):
#         self.gen_npoint_test(29)
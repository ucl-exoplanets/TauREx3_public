"""Functions related to computing emission spectrums"""

import scipy.constants as con
import numpy as np
import ctypes as C
from scipy.stats.mstats_basic import tmean
import numba
import math
from numba import vectorize, float64
from taurex.constants import PI,PLANCK, SPDLIGT, KBOLTZ

@numba.vectorize([float64(float64)],fastmath=True)
def _convert_lamb(lamb):
    return 10000*1e-6/lamb

@numba.vectorize([float64(float64,float64)],fastmath=True)
def _black_body_vec(wl,temp):
    return (PI* (2.0*PLANCK*SPDLIGT**2)/(wl)**5) * (1.0/(np.exp((PLANCK * SPDLIGT) / (wl * KBOLTZ * temp))-1))*1e-6

@numba.njit
def black_body(lamb,temp):
    
    res = np.empty(lamb.shape, dtype=lamb.dtype)
    wl = _convert_lamb(lamb)
#    for i in range(lamb.shape[0]):
#
#        res[i] = (PI* (2.0*PLANCK*SPDLIGT**2)/(wl[i])**5) * (1.0/(math.exp((PLANCK * SPDLIGT) / (wl[i] * KBOLTZ * temp))-1))*1e-6
    
    
    return _black_body_vec(wl,temp)


def black_body_numexpr(lamb, temp):
    import numexpr as ne
    wl = ne.evaluate('10000*1e-6/lamb')
    
    return ne.evaluate('(PI* (2.0*PLANCK*SPDLIGT**2)/(wl)**5) * (1.0/(exp((PLANCK * SPDLIGT) / (wl * KBOLTZ * temp))-1))*1e-6')

def black_body_numpy(lamb, temp):


    h = 6.62606957e-34
    c = 299792458
    k = 1.3806488e-23
    pi= 3.14159265359

    wl = 10000/lamb

    exponent = np.exp((h * c) / (wl*1e-6 * k * temp))
    BB = (pi* (2.0*h*c**2)/(wl*1e-6)**5) * (1.0/(exponent -1))
    return BB * 1e-6

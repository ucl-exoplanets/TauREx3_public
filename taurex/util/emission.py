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

@numba.njit(fastmath=True, parallel=False)
def black_body_numba(lamb,temp):
    
    res = np.empty_like(lamb)
    wl = _convert_lamb(lamb)
#    for i in range(lamb.shape[0]):
#
#        res[i] = (PI* (2.0*PLANCK*SPDLIGT**2)/(wl[i])**5) * (1.0/(math.exp((PLANCK * SPDLIGT) / (wl[i] * KBOLTZ * temp))-1))*1e-6
    
    
    return _black_body_vec(wl,temp)

@numba.njit(fastmath=True,parallel=False)
def black_body_numba_II(lamb, temp):
    N = lamb.shape[0]
    out = np.zeros_like(lamb)
    conversion = 10000*1e-6
    # for n in range(N):
    #     wl[n] = 10000*1e-6/lamb[n]
    
    factor = PI*(2.0*PLANCK*SPDLIGT**2)*1e-6/conversion**5
    c2 = PLANCK * SPDLIGT/(KBOLTZ*temp)/conversion
    
    for n in range(N):
        out[n] = factor*lamb[n]**5/(math.exp(c2*lamb[n])-1)

    return out


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



def integrate_emission_layer(dtau, layer_tau, mu, BB):
    _mu = 1/mu[:,None]
    _tau = np.exp(-layer_tau) - np.exp(-dtau)

    return BB*(np.exp(-layer_tau*_mu) - np.exp(-dtau*_mu)), _tau

@numba.njit(fastmath=True)
def integrate_emission_numba(wngrid, dtau, layer_tau, mu, T):
    n_mu = mu.shape[0]
    nlayers = T.shape[0]
    num_grid = dtau.shape[-1]
    tau = np.zeros(shape=(nlayers, num_grid))
    BB= np.zeros(shape=(nlayers, num_grid))
    for n in range(nlayers):
        BB[n] = black_body_numba_II(wngrid, T[n])

    I = np.zeros(shape=(n_mu, nlayers, num_grid))

    for l in range(nlayers):
        for n in range(num_grid):
            tau[l, n] = np.exp(layer_tau[l, n]) - np.exp(dtau[l, n])

    for m in range(n_mu):
        _mu = 1/mu[m]
        for l in range(nlayers):
            for n in range(num_grid):
                I[m,l,n] = BB[l,n]*(np.exp(-layer_tau[l, n]*_mu) - np.exp(-dtau[l, n]*_mu))
    
    return I, tau






black_body = black_body_numba
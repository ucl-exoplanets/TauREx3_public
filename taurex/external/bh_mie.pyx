
# -*- mode: python -*-
#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
# Comments above are special. Please do not remove.
cimport numpy as np  # needed for function arguments
import numpy as np # needed for np.empty_like


ctypedef np.float32_t float_t
ctypedef np.float64_t double_t
ctypedef np.int32_t int_t

cdef extern from "bhmie_lib.h":
    void compute_sigma_mie(
            const double a, 
            const int nwgrid,   
            const double * wavegrid,
            const double * ref_real, 
            const double * ref_imag,
            void * sigma_out        
            );

def bh_mie(double particle_radius, 
          np.ndarray[dtype=double_t, ndim=1, mode="c"] wavegrid,
          np.ndarray[dtype=double_t, ndim=1, mode="c"] ref_real,
          np.ndarray[dtype=double_t, ndim=1, mode="c"] ref_imag,
          ):

    cdef np.ndarray[dtype=double_t, ndim=1, mode="c"] output
    output = np.empty_like(wavegrid, dtype='d')

    if not (wavegrid.shape[0] == ref_real.shape[0]):
        raise ValueError("wave grid and ref_real shapes are not consistent")

    if not (ref_real.shape[0] == ref_imag.shape[0]):
        raise ValueError("ref_real and ref_imag shapes are not consistent")

    compute_sigma_mie(particle_radius, wavegrid.shape[0],<double*>wavegrid.data, <double*>ref_real.data,<double*>ref_imag.data, <double*>output.data)

    return output
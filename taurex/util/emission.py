import scipy.constants as con
import numpy as np
import ctypes as C
from scipy.stats.mstats_basic import tmean

def black_body(lamb, temp):
    #small function calculating plank black body
    #input: WN, kelvin
    #output: W/m^2/micron
    
    h = 6.62606957e-34
    c = 299792458
    k = 1.3806488e-23
    pi= 3.14159265359
    
    wl = 10000/lamb

    exponent = np.exp((h * c) / (wl*1e-6 * k * temp))
    BB = (pi* (2.0*h*c**2)/(wl*1e-6)**5) * (1.0/(exponent -1))

#            exponent = exp((h * c) / ((10000./wngrid[wn])*1e-6  * kb * temperature[0]));
#            BB_wl = ((2.0*h*pow(c,2))/pow((10000./wngrid[wn])*1e-6,5) * (1.0/(exponent - 1)))* 1e-6; // (W/m^2/micron)

#     exponent = np.exp((con.h * con.c) / (lamb *1e-6 * con.k * temp))
#     BB = (np.pi* (2.0*con.h*con.c**2)/(lamb*1e-6)**5) * (1.0/(exponent -1))
    
    return BB * 1e-6
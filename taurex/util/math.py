"""Optimized Math functions used in taurex"""

import numexpr as ne
import numpy as np



def interp_exp_and_lin(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
        """
        2D interpolation

        Applies linear interpolation across P and natural exp interpolation
        across T between Pmin,Pmax and Tmin,Tmax

        Parameters
        ----------
        x11: array
            Array corresponding to Pmin,Tmin
        
        x12: array
            Array corresponding to Pmin,Tmax
        
        x21: array 
            Array corresponding to Pmax,Tmin
        
        x22: array
            Array corresponding to Pmax,Tmax

        T: float
            Coordinate to exp interpolate to

        Tmin: float
            Nearest known T coordinate where Tmin < T
        
        Tmax: float
            Nearest known T coordinate where T < Tmax

        P: float
            Coordinate to linear interpolate to

        Pmin: float
            Nearest known P coordinate where Pmin < P
        
        Pmax: float
            Nearest known P coordinate where P < Tmax
        
        """

        return ne.evaluate('((x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x21))*exp(Tmax*(-T + Tmin)*log((x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x21))/(x12*(Pmax - Pmin) - (P - Pmin)*(x12 - x22)))/(T*(Tmax - Tmin)))/(Pmax - Pmin))')


def interp_exp_only(x11,x12,T,Tmin,Tmax):
    return ne.evaluate('x11*exp(Tmax*(-T + Tmin)*log(x11/x12)/(T*(Tmax - Tmin)))')

def interp_lin_only(x11,x12,P,Pmin,Pmax):
    return ne.evaluate('(x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x12))/(Pmax - Pmin)')


def intepr_bilin(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    return ne.evaluate('(x11*(Pmax - Pmin)*(Tmax - Tmin) - (P - Pmin)*(Tmax - Tmin)*(x11 - x21) - (T - Tmin)*(-(P - Pmin)*(x11 - x21) + (P - Pmin)*(x12 - x22) + (Pmax - Pmin)*(x11 - x12)))/((Pmax - Pmin)*(Tmax - Tmin))')


def compute_rayleigh_cross_section(wngrid,n,n_air = 2.6867805e25,king=1.0):
    wlgrid = (10000/wngrid)*1e-6

    
    n_factor = (n**2 - 1)/(n_air*(n**2 + 2))
    sigma = 24.0*(np.pi**3)*king*(n_factor**2)/(wlgrid**4)

    return sigma




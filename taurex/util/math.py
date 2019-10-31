"""Optimized Math functions used in taurex"""

import numba
import numpy as np

from numba import vectorize, float64
import math

@numba.vectorize([float64(float64,float64,float64, float64)])
def _expstage0(x1,x2, x11, x21):
    return x1*x11 - x2*(x11-x21)
@numba.vectorize([float64(float64,float64)],fastmath=True)
def _expstage1(x1, x2):
    return math.log(x1/x2)

@numba.vectorize([float64(float64,float64)],fastmath=True)
def _expstage2(C,x):
    return C*x

@numba.vectorize([float64(float64,float64,float64)],fastmath=True)
def _expstage3(C,x1,x2):
    return C*x1*x2

@numba.njit(nogil=True,fastmath=True)
def interp_exp_and_lin_broken(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    res = np.zeros_like(x11)
    x0 = -Pmin
    x1 = Pmax + x0
    x2 = P + x0
    factor1 = 1.0/(T*(Tmax - Tmin))
    factor2 = 1.0/x1
    x3 = _expstage0(x1,x2,x11,x21)
    x4 = _expstage0(x1,x2,x12,x22)
    x5 = _expstage1(x3,x4)
    x6 = _expstage2(Tmax*(-T + Tmin)*factor1,x5)
    x7 = _expstage3(factor2,x3,x6)
    for i in range(x11.shape[0]):
        res[i] = x7[i]*math.exp(x6[i])
    return res 


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
    return x11*np.exp(Tmax*(-T + Tmin)*np.log(x11/x12)/(T*(Tmax - Tmin)))

def interp_lin_only(x11,x12,P,Pmin,Pmax):
    return (x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x12))/(Pmax - Pmin)


def intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    return (x11*(Pmax - Pmin)*(Tmax - Tmin) - (P - Pmin)*(Tmax - Tmin)*(x11 - x21) - (T - Tmin)*(-(P - Pmin)*(x11 - x21) + (P - Pmin)*(x12 - x22) + (Pmax - Pmin)*(x11 - x12)))/((Pmax - Pmin)*(Tmax - Tmin))


def compute_rayleigh_cross_section(wngrid,n,n_air = 2.6867805e25,king=1.0):
    wlgrid = (10000/wngrid)*1e-6

    
    n_factor = (n**2 - 1)/(n_air*(n**2 + 2))
    sigma = 24.0*(np.pi**3)*king*(n_factor**2)/(wlgrid**4)

    return sigma

def test_nan(val):
    if hasattr(val,'__len__'):
        try:
            return np.isnan(val).any()
        except TypeError:
           # print(type(val))
            return True
    else:
        return val != val


@numba.vectorize([float64(float64,float64,float64)],fastmath=True)
def _linstage0(x11,x21,x):
    return x*(x11-x21)

@numba.njit(nogil=True,fastmath=True)
def intepr_bilin(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    x0 = -Pmin    
    x1 = Pmax + x0
    x2 = -Tmin    
    x3 = Tmax + x2
    x4 = P + x0    
    
    factor = 1.0/(x1*x3)
    
    x5 = _linstage0(x11,x21,x4)
    x6 = _linstage0(x11,x12,x1)
    x7 = _linstage0(x12,x22,x4)
    res = np.zeros_like(x11)
    for i in range(x11.shape[0]):
        res[i] = (x1*x11[i]*x3 - x3*x5[i] - (T + x2)*(x6[i] + x7[i] - x5[i]))*factor
    
    return res


class OnlineVariance(object):
    """USes the M2 algorithm to compute the variance in a streaming fashion"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.wcount = 0.0
        self.wcount2=0.0
        self.mean = None
        self.M2 = None
       
    
    def update(self,value,weight=1.0):
        self.count+=1
        self.wcount+=weight
        self.wcount2+=weight*weight


        if self.mean is None:
            self.mean = value*0.0
            self.M2 = value*0.0
        

        mean_old = self.mean
        try:
            self.mean = mean_old + (weight / self.wcount) * (value - mean_old)
        except ZeroDivisionError:
            self.mean = value*0.0
        self.M2 += weight * (value - mean_old) * (value - self.mean)

    @property
    def variance(self):
        if self.count < 2:
            return np.nan
        else:
            return self.M2/self.wcount
    
    @property
    def sampleVariance(self):
        if self.count < 2:
            return np.nan
        else:
            return self.M2/(self.wcount-1)


    # def combine_variance(self,averages, variances, counts):
    #     good_idx = [idx for idx,a in enumerate(averages) if not test_nan(a)]
    #     averages = [averages[idx] for idx in good_idx]
    #     variances = [variances[idx] for idx in good_idx]
    #     counts = [counts[idx] for idx in good_idx]
    #     good_variance = None
    #     if not test_nan(variances):
    #         try:
    #             good_variance = variances[np.where(~np.isnan(variances))[0][0]]*0.0
    #         except IndexError:
    #             good_variance = None
    #     #print(good_idx,'Good',good_variance)
    #     variances = [v if not test_nan(v) else good_variance for v in variances] 
    #     #print('NEWAVERAGES',averages) 
    #     #print('NEW WEIGHTS',counts)      
        
    #     average = np.average(averages, weights=counts,axis=0)
        
    #     #print('final average',average)
    #     size = np.sum(counts)
        
    #     counts = np.array(counts) * size/np.sum(counts)
    #     if hasattr(average,'__len__'):
    #         average = average[None,...]
    #         for x in range(1,len(average.shape)):
    #             counts = counts[:,None]
    #     squares = 0.0
    #     if good_variance is not None:
    #         squares = counts*np.nan_to_num(variances)
    #     #print(counts,variances,squares)
    #     squares = squares + counts*(average - averages)**2

    #     return average,np.sum(squares,axis=0)/size

    def combine_variance(self, averages, variance, counts):
        average = None
        size = np.sum(counts)
        for avg,cnt in zip(averages,counts):
            if cnt == 0:
                continue

            #print('avg',avg)
            if avg is not None and not avg is np.nan:
                if average is None:
                    average = avg*cnt
                else:
                    average += avg*cnt
        average/=size
        #print('AVERGAE',average)
        counts = np.array(counts) * size/np.sum(counts)

        squares = None

        for avg,cnt,var in zip(averages,counts,variance):
            #print('COUNT ',cnt)
            if cnt == 0.0:
                continue
            if cnt > 0.0:
                if squares is None:
                    squares = cnt*(average - avg)**2
                else:
                    squares += cnt*(average - avg)**2
            if var is not np.nan:
                squares += cnt*var 
        # squares = counts*variances
        # squares += counts*(average - averages)**2

        return average,squares/size
    def parallelVariance(self):
        from taurex import mpi

        variance = self.variance
        
        mean = self.mean
        if mean is None:
            mean = np.nan


        variances = mpi.allgather(variance)


        averages = mpi.allgather(mean)
        counts = mpi.allgather(self.wcount)
        #all_data = [(v,m,c) for v,m,c in zip(variances,averages,counts) if not v is 0 and not averages is 0.0 and not counts is 0.0]        
        #print('VARIANCES',variances)
        #print('AVERAGES',averages)
        #print('COUNTS',counts)
        
        finalvariance = self.combine_variance(averages,variances,counts)
        return finalvariance[-1]

        
        

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

def test_nan(val):
    if hasattr(val,'__len__'):
        return np.isnan(val).any()
    else:
        return val != val
        


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


    def combine_variance(self,averages, variances, counts):
        good_idx = [idx for idx,a in enumerate(averages) if not test_nan(a)]
        averages = [averages[idx] for idx in good_idx]
        variances = [variances[idx] for idx in good_idx]
        counts = [counts[idx] for idx in good_idx]
        good_variance = None
        if not test_nan(variances):
            try:
                good_variance = variances[np.where(~np.isnan(variances))[0][0]]*0.0
            except IndexError:
                good_variance = None
        #print(good_idx,'Good',good_variance)
        variances = [v if not test_nan(v) else good_variance for v in variances] 
        #print('NEWAVERAGES',averages) 
        #print('NEW WEIGHTS',counts)      
        
        average = np.average(averages, weights=counts,axis=0)
        
        #print('final average',average)
        size = np.sum(counts)
        
        counts = np.array(counts) * size/np.sum(counts)
        if hasattr(average,'__len__'):
            average = average[None,...]
            for x in range(1,len(average.shape)):
                counts = counts[:,None]
        squares = 0.0
        if good_variance is not None:
            squares = counts*np.nan_to_num(variances)
        #print(counts,variances,squares)
        squares = squares + counts*(average - averages)**2

        return average,np.sum(squares,axis=0)/size
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

        
        

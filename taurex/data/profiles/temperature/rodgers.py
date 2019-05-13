from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.util import movingaverage
class Rodgers2000(TemperatureProfile):
    """
    """

    def __init__(self,temperature_layers=[],correlation_length=5.0,covariance_matrix=None):
        super().__init__('Rodgers2000')

        self._tp_corr_length = correlation_length
        self._covariance = covariance_matrix
        self._T_layers = np.array(temperature_layers)
        self.generate_temperature_fitting_params()

    def gen_covariance(self):
        h = self._tp_corr_length
        pres_prof = (self.pressure_profile)

        return np.exp(-1.0*np.abs(np.log(pres_prof[:,None]/pres_prof[None,:])) /h)
    

    def correlate_temp(self,cov_mat):

        cov_mat_sum = np.sum(cov_mat,axis=0)
        weights = cov_mat[:,:]/cov_mat_sum[:,None]  
        return weights.dot(self._T_layers)
    
    @property
    def profile(self):

        cov_mat = self._covariance
        if cov_mat is None:
            cov_mat = self.gen_covariance()


        return self.correlate_temp(cov_mat)

    @fitparam(param_name='corr_length',param_latex='$C_\{L\}$',default_fit=False,default_bounds=[1.0,10.0])
    def correlationLength(self):
        return self._tp_corr_length
    
    @correlationLength.setter
    def correlationLength(self,value):
        self._tp_corr_length = value

    def generate_temperature_fitting_params(self):


        bounds = [1e5,1e3]
        for idx,val in enumerate(self._T_layers):
            point_num = idx+1
            param_name = 'T_{}'.format(point_num)
            param_latex = '$T_{%i}$' % point_num
            def read_point(self,idx=idx):
                return self._T_layers[idx]
            def write_point(self,value,idx=idx):
                self._T_layers[idx] = value

            fget_point = read_point
            fset_point = write_point
            default_fit = False
            self.add_fittable_param(param_name,param_latex ,fget_point,fset_point,default_fit,bounds) 
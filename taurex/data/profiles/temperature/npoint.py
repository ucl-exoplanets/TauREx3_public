from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.util import movingaverage
class NPoint(TemperatureProfile):
    """

    TP profile from Guillot 2010, A&A, 520, A27 (equation 49)
    Using modified 2stream approx. from Line et al. 2012, ApJ, 749,93 (equation 19)


    Parameters
    -----------
        T_irr: :obj:`float` 
            planet equilibrium temperature (Line fixes this but we keep as free parameter)
        kappa_ir: :obj:`float`
            mean infra-red opacity
        kappa_v1: :obj:`float` 
            mean optical opacity one
        kappa_v2: :obj:`float` 
            mean optical opacity two
        alpha: :obj:`float` 
            ratio between kappa_v1 and kappa_v2 downwards radiation stream

    """


    def __init__(self,T_surface=100.0,T_top=20.0,P_surface=None,P_top=None,temperature_points=[],pressure_points=[],smoothing_window=10):
        super().__init__('{}Point'.format(len(temperature_points)+1))

        if not isinstance(temperature_points,list):
            raise Exception('t_point is not a list')

        if len(temperature_points) != len(pressure_points):
            self.error('Number of temeprature points != number of pressure points')
            self.error('len(t_points) = {} /= len(p_points) = {}'.format(len(temperature_points),len(pressure_points)))
            raise Exception('Incorrect_number of temp and pressure points')
        
        self.info('Npoint temeprature profile is initialized')
        self.debug('Passed temeprature points {}'.format(temperature_points))
        self.debug('Passed pressure points {}'.format(pressure_points))
        self._t_points = temperature_points
        self._p_points = pressure_points
        self._T_surface = T_surface
        self._T_top = T_top
        self._P_surface = P_surface
        self._P_top = P_top
        self._smooth_window = smoothing_window
        self.generate_pressure_fitting_params()
        self.generate_temperature_fitting_params()
    @fitparam(param_name='T_surface',param_latex='$T_\\mathrm{surf}$',default_fit=False,default_bounds=[ 300,2500])
    def temperatureSurface(self):
        return self._T_surface
    
    @temperatureSurface.setter
    def temperatureSurface(self,value):
        self._T_surface = value

    @fitparam(param_name='T_top',param_latex='$T_\\mathrm{top}$',default_fit=False,default_bounds=[ 300,2500])
    def temperatureTop(self):
        return self._T_top
    
    @temperatureTop.setter
    def temperatureTop(self,value):
        self._T_top = value

    @fitparam(param_name='P_surface',param_latex='$P_\\mathrm{surf}$',default_fit=False,default_bounds=[ 1e3,1e2])
    def pressureSurface(self):
        return self._P_surface
    
    @pressureSurface.setter
    def pressureSurface(self,value):
        self._P_surface = value

    @fitparam(param_name='P_top',param_latex='$P_\\mathrm{top}$',default_fit=False,default_bounds=[ 1e-5,1e-4])
    def pressureTop(self):
        return self._P_top
    
    @pressureTop.setter
    def pressureTop(self,value):
        self._P_top = value
 

    def generate_pressure_fitting_params(self):


        bounds = [1e5,1e3]
        for idx,val in enumerate(self._p_points):
            point_num = idx+1
            param_name = 'P_point{}'.format(point_num)
            param_latex = '$P_{}$'.format(point_num)
            def read_point(self,idx=idx):
                return self._p_points[idx]
            def write_point(self,value,idx=idx):
                self._p_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location {} {}'.format(fget_point,fget_point(self)))
            default_fit = False
            self.add_fittable_param(param_name,param_latex ,fget_point,fset_point,default_fit,bounds) 



    def generate_temperature_fitting_params(self):


        bounds = [300,2500]
        for idx,val in enumerate(self._t_points):
            point_num = idx+1
            param_name = 'T_point{}'.format(point_num)
            param_latex = '$T_{}$'.format(point_num)
            
            def read_point(self,idx=idx):
                return self._t_points[idx]
            def write_point(self,value,idx=idx):
                self._t_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location {} {}'.format(fget_point,fget_point(self)))
            default_fit = False
            self.add_fittable_param(param_name,param_latex ,fget_point,fset_point,default_fit,bounds)        





    @property
    def profile(self):

        Tnodes = [self._T_surface,*self._t_points,self._T_top]
        
        Psurface = self._P_surface
        if Psurface is None:
            Psurface = self.pressure_profile[0]
        
        Ptop = self._P_top
        if Ptop is None:
            Ptop = self.pressure_profile[-1] 
        
        Pnodes = [Psurface,*self._p_points,Ptop]

        smooth_window = self._smooth_window


        TP = np.interp((np.log(self.pressure_profile[::-1])), np.log(Pnodes[::-1]), Tnodes[::-1])
        #smoothing T-P profile
        wsize = int(self.nlayers*(smooth_window/100.0))
        if (wsize %2 == 0):
            wsize += 1
        TP_smooth = movingaverage(TP,wsize)
        border = np.int((len(TP) - len(TP_smooth))/2)

        #set atmosphere object
        foo = TP[::-1]
        foo[border:-border] = TP_smooth[::-1]

        return foo


    def write(self,output):
        temperature = super().write(output)

        temperature.write_scalar('temp_surface',self._T_surface)
        temperature.write_scalar('temp_top',self._T_top)
        temperature.write_array('temp_points',self._t_points)

        P_surface = self._P_surface
        P_top = self._P_top
        if not P_surface:
            P_surface = -1
        if not P_top:
            P_top = -1

        temperature.write_scalar('pressure_surface',P_surface)
        temperature.write_scalar('pressure_top',P_top)
        temperature.write_array('pressure_points',self._p_points)

        temperature.write_scalar('smoothin_window',self._smooth_window)

        return temperature
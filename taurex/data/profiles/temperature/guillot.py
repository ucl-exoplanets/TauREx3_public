from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.exceptions import InvalidModelException

class Guillot2010(TemperatureProfile):
    """

    TP profile from Guillot 2010, A&A, 520, A27 (equation 49)
    Using modified 2stream approx.
    from Line et al. 2012, ApJ, 749,93 (equation 19)


    Parameters
    -----------
        T_irr: float
            planet equilibrium temperature
            (Line fixes this but we keep as free parameter)
        kappa_ir: float
            mean infra-red opacity
        kappa_v1: float
            mean optical opacity one
        kappa_v2: float
            mean optical opacity two
        alpha: float
            ratio between kappa_v1 and kappa_v2 downwards radiation stream
        T_int: float
            Internal heating parameter (K)

    """

    def __init__(self, T_irr=1500, kappa_irr=0.01, kappa_v1=0.005,
                 kappa_v2=0.005, alpha=0.5, T_int=100):
        super().__init__('Guillot')

        self.T_irr = T_irr

        self.kappa_ir = kappa_irr
        self.kappa_v1 = kappa_v1
        self.kappa_v2 = kappa_v2
        self.alpha = alpha
        self.T_int = T_int

        self._check_values()

    @fitparam(param_name='T_irr',
              param_latex='$T_\\mathrm{irr}$',
              default_fit=True,
              default_bounds=[1300, 2500])
    def equilTemperature(self):
        """Planet equilibrium temperature"""
        return self.T_irr

    @equilTemperature.setter
    def equilTemperature(self, value):
        self.T_irr = value

    @fitparam(param_name='kappa_irr',
              param_latex='$k_\\mathrm{irr}$',
              default_fit=False,
              default_bounds=[1e-10, 1],
              default_mode='log')
    def meanInfraOpacity(self):
        """mean infra-red opacity"""
        return self.kappa_ir

    @meanInfraOpacity.setter
    def meanInfraOpacity(self, value):
        self.kappa_ir = value

    @fitparam(param_name='kappa_v1',
              param_latex='$k_\\mathrm{1}$',
              default_fit=False,
              default_bounds=[1e-10, 1],
              default_mode='log')
    def meanOpticalOpacity1(self):
        """mean optical opacity one"""
        return self.kappa_v1

    @meanOpticalOpacity1.setter
    def meanOpticalOpacity1(self, value):
        self.kappa_v1 = value

    @fitparam(param_name='kappa_v2',
              param_latex='$k_\\mathrm{2}$',
              default_fit=False,
              default_bounds=[1e-10, 1],
              default_mode='log')
    def meanOpticalOpacity2(self):
        """mean optical opacity two"""
        return self.kappa_v2

    @meanOpticalOpacity2.setter
    def meanOpticalOpacity2(self, value):
        self.kappa_v2 = value

    @fitparam(param_name='alpha', param_latex='$\\alpha$',
              default_fit=False, default_bounds=[0.0, 1.0])
    def opticalRatio(self):
        """ratio between kappa_v1 and kappa_v2 """
        return self.alpha

    @opticalRatio.setter
    def opticalRatio(self, value):
        self.alpha = value

    @fitparam(param_name='T_int_guillot', param_latex='$T^{g}_{int}$',
              default_fit=False, default_bounds=[0.0, 1.0])
    def internalTemperature(self):
        """ratio between kappa_v1 and kappa_v2 """
        return self.T_int

    @internalTemperature.setter
    def internalTemperature(self, value):
        self.T_int = value

    def _check_values(self):
        """
        Ensures kappa values are valid

        Raises
        ------
        InvalidModelException:
            If any kappa is zero
        
        """

        if self.kappa_ir == 0.0:
            self.warning('Kappa ir is zero')
            raise InvalidModelException('kappa_ir is zero')

        gamma_1 = self.kappa_v1/(self.kappa_ir)
        gamma_2 = self.kappa_v2/(self.kappa_ir)

        if gamma_1 == 0.0 or gamma_2 == 0.0:
            self.warning('Gamma is zero. kappa_v1 = %s kappa_v2 = %s'
                         ' kappa_ir = %s',
                         self.kappa_v1,
                         self.kappa_v2, self.kappa_ir)
            raise InvalidModelException('Kappa v1/v2/ir values result in zero gamma')
        
        if self.T_irr < 0 or self.T_int < 0:
            self.warning('Negative temperature input T_irr=%s T_int=%s',
                         self.T_irr, self.T_int)
            raise InvalidModelException('Negative temperature input')   


    @property
    def profile(self):
        """

        Returns a guillot temperature temperature profile

        Returns
        --------

        temperature_profile : :obj:np.array(float)

        """

        planet_grav = self.planet.gravity

        self._check_values()

        gamma_1 = self.kappa_v1/(self.kappa_ir)
        gamma_2 = self.kappa_v2/(self.kappa_ir)
        tau = self.kappa_ir * self.pressure_profile / planet_grav


        T_int = self.T_int  # todo internal heat parameter looks suspicious..

        def eta(gamma, tau):
            import scipy.special as spe

            part1 = 2.0/3.0 + 2.0 / (3.0*gamma) * (1.0 +
                                                   (gamma*tau/2.0 - 1.0) *
                                                   np.exp(-1.0 * gamma * tau))

            part2 = 2.0 * gamma / 3.0 * (1.0 - tau**2/2.0) * \
                spe.expn(2, (gamma * tau))

            return part1 + part2

        T4 = 3.0 * T_int**4/4.0 * (2.0/3.0 + tau) + 3.0*self.T_irr**4 / 4.0 * \
            (1.0 - self.alpha) * eta(gamma_1, tau) + \
            3.0 * self.T_irr**4/4.0 * self.alpha * eta(gamma_2, tau)

        T = T4**0.25

        return T

    def write(self, output):
        temperature = super().write(output)
        temperature.write_scalar('T_irr', self.T_irr)
        temperature.write_scalar('kappa_irr', self.kappa_ir)
        temperature.write_scalar('kappa_v1', self.kappa_v1)
        temperature.write_scalar('kappa_v2', self.kappa_v2)
        temperature.write_scalar('alpha', self.alpha)
        return temperature

    @classmethod
    def input_keywords(cls):
        """
        Return all input keywords
        """
        return ['guillot', 'guillot2010', ]

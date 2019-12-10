from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam


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

    """

    def __init__(self, T_irr=1500, kappa_irr=0.01, kappa_v1=0.005,
                 kappa_v2=0.005, alpha=0.5):
        super().__init__('Guillot')

        self.T_irr = T_irr

        self.kappa_ir = kappa_irr
        self.kappa_v1 = kappa_v1
        self.kappa_v2 = kappa_v2
        self.alpha = alpha

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

    @property
    def profile(self):
        """

        Returns a guillot temperature temperature profile

        Returns
        --------

        temperature_profile : :obj:np.array(float)

        """

        planet_grav = self.planet.gravity
        gamma_1 = self.kappa_v1/(self.kappa_ir + 1e-10)
        gamma_2 = self.kappa_v2/(self.kappa_ir + 1e-10)
        tau = self.kappa_ir * self.pressure_profile / planet_grav

        T_int = 100  # todo internal heat parameter looks suspicious..

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

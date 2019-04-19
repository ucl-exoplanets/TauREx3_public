from .tprofile import TemperatureProfile
import numpy as np


class Guillot2010(TemperatureProfile):
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


    def __init__(self,T_irr,kappa_irr,kappa_v1,kappa_v2,alpha):
        super().__init__('Guillot')

        self.T_irr = T_irr
        self.kappa_ir = np.power(10, kappa_irr)
        self.kappa_v1 = np.power(10, kappa_v1)
        self.kappa_v2 = np.power(10, kappa_v2)
        self.alpha = alpha
    
    def profile(self):
        """Returns an isothermal temperature profile

        Returns
        --------
        
        :obj:np.array(float)
            temperature profile
        """

        planet_grav = self.planet.gravity
        gamma_1 = self.kappa_v1/(self.kappa_ir + 1e-10); gamma_2 = self.kappa_v2/(self.kappa_ir + 1e-10)
        tau = self.kappa_ir * self.pressure_profile / planet_grav


        T_int = 100 # todo internal heat parameter looks suspicious... needs looking at.

        def eta(gamma, tau):
            import scipy.special as spe
            part1 = 2.0/3.0 + 2.0 / (3.0*gamma) * (1.0 + (gamma*tau/2.0 - 1.0) * np.exp(-1.0 * gamma * tau))
            part2 = 2.0 * gamma / 3.0 * (1.0 - tau**2/2.0) * spe.expn(2,(gamma*tau))
            return part1 + part2

        T4 = 3.0*T_int**4/4.0 * (2.0/3.0 + tau) + 3.0*self.T_irr**4/4.0 *(1.0 - self.alpha) * eta(gamma_1,tau) + 3.0 * self.T_irr**4/4.0 * self.alpha * eta(gamma_2,tau)
        T = T4**0.25

        return T


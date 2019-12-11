 
from .contribution import Contribution, contribute_tau
import numpy as np
from taurex.data.fittable import fitparam



class HydrogenIon(Contribution):


    def __init__(self):
        super().__init__('HydrogenIon')


    def build(self, model):
        pass

    def finalize(self, model):
        raise NotImplementedError

    def calculate_bb(self, wlg, T):
        k_H_min = np.zeros(len(wlg))

        ### An Bn Cn Dn En Fn
        coeffs1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520],
                   [-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390],
                   [2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380],
                   [-696.2710, -1841.4000, 8642.9700, -10051.5300, 4400.0670, -655.0200],
                   [88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850]]
        coeffs2 = [[518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
                   [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
                   [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
                   [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        for i, wl in enumerate(wlg):
            if wl > 0.3645:
                Xn = coeffs1
            elif wl > 0.1823 and wl <= 0.3645:
                Xn = coeffs2
            else:
                k_H_min[i] = 0
                continue
            sum = 0.0
            for n in range(6):
                sum = sum + (5040.0 / T) ** (float(n + 1) / 2.0) * \
                      (wl ** 2 * Xn[n, 0] + Xn[n, 1] + Xn[n, 2] / wl + Xn[n, 3] / wl ** 2 + Xn[n, 4] / wl ** 3 + Xn[
                          n, 5] / wl ** 4)
            k_H_min[i] = 1.0e-29 * sum
        return k_H_min

    def calculate_bf(self, wlg, T):
        wl0 = 1.6419
        alp = 1.43910 ** 8
        Cn = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
        k_H_min = np.zeros(len(wlg))

        for i, wl in enumerate(wlg):
            if wl >= 0.125 and wl <= 1.6419:

                sum = 0.0
                for n in range(6):
                    sum = sum + Cn[n] * (1.0 / wl - 1.0 / wl0) ** (float(n) / 2.0)

                xsec_H_min = 1.0e-18 * wl ** 3 * (1.0 / wl - 1.0 / wl0) ** (3.0 / 2.0) * sum

                k_H_min[i] = 0.750 * T ** (-5.0 / 2.0) * np.exp(alp / (wl0 * T)) * (
                            1.0 - np.exp(-alp / (wl * T))) * xsec_H_min

            elif wl > 1.6419:
                k_H_min[i] = 0.0

        return k_H_min


    def prepare_each(self, model, wngrid):
        self._nlayers = model.nLayers
        self._P_dyne = model.pressureProfile * 1e6
        self._temperature_profile = model.temperatureProfile


        self._hydrogen_mixratio = model.chemistry.get_gas_mix_profile('H')
        self.debug('hydrogen %s', self._hydrogen_mixratio)
        self._electron_mixratio = model.chemistry.get_gas_mix_profile('e-')
        self.debug('e- %s', self._electron_mixratio)

        xsec_bb = []
        xsec_bf = []

        for i in range(self._nlayers):
            xsec_bb = self.calculate_bb(10000 / wngrid, self._temperature_profile[i]) * self._P_dyne[i] * \
                      self._hydrogen_mixratio[i] * self._electron_mixratio[i]
            xsec_bf = self.calculate_bf(10000 / wngrid, self._temperature_profile[i]) * self._P_dyne[i] * \
                      self._hydrogen_mixratio[i] * self._electron_mixratio[i]

        self.sigma_xsec = xsec_bb + xsec_bf

        self.debug('final xsec %s', self.sigma_xsec[:, :])
        self.debug('final xsec %s', self.sigma_xsec.max())
        
        #self._total_contrib[...]=0.0
        yield 'HydrogenIon', self.sigma_xsec

    @property
    def sigma(self):
        return self.sigma_xsec

    def write(self, output):
        raise NotImplementedError
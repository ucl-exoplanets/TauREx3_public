from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam


class Rodgers2000(TemperatureProfile):
    """
    Layer-by-layer temperature - pressure profile retrieval using dampening
    factor Introduced in Rodgers (2000): Inverse Methods for Atmospheric
    Sounding (equation 3.26). Featured in NEMESIS code (Irwin et al., 2008,
    J. Quant. Spec., 109, 1136 (equation 19)
    Used in all Barstow et al. papers.

    Parameters
    ----------
    temperature_layers : :obj:`list`
        Temperature in Kelvin per layer of pressure

    correlation_length : float
        In scaleheights, Line et al. 2013 sets this to 7, Irwin et al sets
        this to 1.5 may be left as free and Pressure dependent parameter later.

    covariance_matrix : :obj:`array` , optional
        User can supply their own covaraince matrix

    """

    def __init__(self, temperature_layers=[], correlation_length=5.0,
                 covariance_matrix=None):
        super().__init__('Rodgers2000')

        self._tp_corr_length = correlation_length
        self._covariance = covariance_matrix
        self._T_layers = np.array(temperature_layers)
        self.generate_temperature_fitting_params()

    def gen_covariance(self):
        """
        Generate the covariance matrix if None is supplied
        """
        h = self._tp_corr_length
        pres_prof = (self.pressure_profile)

        return np.exp(-1.0 * np.abs(np.log(pres_prof[:, None] /
                                           pres_prof[None, :])) / h)

    def correlate_temp(self, cov_mat):
        cov_mat_sum = np.sum(cov_mat, axis=0)
        weights = cov_mat[:, :]/cov_mat_sum[:, None]
        return weights.dot(self._T_layers)

    @property
    def profile(self):

        cov_mat = self._covariance
        if cov_mat is None:
            cov_mat = self.gen_covariance()

        return self.correlate_temp(cov_mat)

    @fitparam(param_name='correlation_length',
              param_latex='$C_{L}$',
              default_fit=False,
              default_bounds=[1.0, 10.0])
    def correlationLength(self):
        """
        Correlation length in scale heights
        """
        return self._tp_corr_length

    @correlationLength.setter
    def correlationLength(self, value):
        self._tp_corr_length = value

    def generate_temperature_fitting_params(self):
        """
        Generates the temperature fitting parameters for each layer of the
        atmosphere For a 4 layer atmosphere the fitting parameters generated
        are ``T_0``, ``T_1``, ``T_2`` and ``T_3``
        """

        bounds = [1e5, 1e3]
        for idx, val in enumerate(self._T_layers):
            point_num = idx+1
            param_name = 'T_{}'.format(point_num)
            param_latex = '$T_{%i}$' % point_num

            def read_point(self, idx=idx):
                return self._T_layers[idx]

            def write_point(self, value, idx=idx):
                self._T_layers[idx] = value

            fget_point = read_point
            fset_point = write_point
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'linear', default_fit, bounds)

    def write(self, output):
        temperature = super().write(output)

        cov_mat = self._covariance
        if cov_mat is None:
            cov_mat = self.gen_covariance()

        temperature.write_array('covariance_matrix', cov_mat)
        temperature.write_array('temperature_layers', self._T_layers)
        temperature.write_scalar('correlation_length', self._tp_corr_length)
        return temperature

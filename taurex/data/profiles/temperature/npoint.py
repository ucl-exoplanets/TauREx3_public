from .tprofile import TemperatureProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.util import movingaverage


class NPoint(TemperatureProfile):
    """

    A temperature profile that is defined at various heights of the
    atmopshere and then smoothend.

    At minimum, temepratures on both the top ``T_top`` and surface
    ``T_surface`` must be defined.
    If any intermediate points are given as ``temperature_points``
    then the same number of ``pressure_points``
    must be given as well.

    A 2-point temperature profile has ``len(temperature_points) == 0``
    A 3-point temperature profile has ``len(temperature_points) == 1``

    etc.


    Parameters
    -----------
        T_surface : float
            Temperature at the planets surface in Kelvin

        T_top : float
            Temperature at the top of the atmosphere in Kelvin

        P_surface : float , optional
            Pressure for ``T_surface`` (Optional) otherwise uses surface
            pressure from forward model

        P_top : float , optional
            Pressure for ``T_top`` (Optional) otherwise uses top pressure from
            forward model

        temperature_points : :obj:`list`
            temperature points between ``T_top`` and ``T_surface``

        pressure_points : :obj:`list`
            pressure points that the each temperature in ``temperature_points``
            lie on

        smoothing_window : int
            smoothing window


    """

    def __init__(self, T_surface=1500.0, T_top=200.0, P_surface=None,
                 P_top=None, temperature_points=[], pressure_points=[],
                 smoothing_window=10):
        super().__init__('{}Point'.format(len(temperature_points)+2))

        if not hasattr(temperature_points, '__len__'):
            raise Exception('t_point is not an iterable')

        if len(temperature_points) != len(pressure_points):
            self.error('Number of temeprature points != number of '
                       'pressure points')
            self.error('len(t_points) = %s /= '
                       'len(p_points) = %s',
                       len(temperature_points),
                       len(pressure_points))
            raise Exception('Incorrect_number of temp and pressure points')

        self.info('Npoint temeprature profile is initialized')
        self.debug('Passed temeprature points %s', temperature_points)
        self.debug('Passed pressure points %s', pressure_points)
        self._t_points = temperature_points
        self._p_points = pressure_points
        self._T_surface = T_surface
        self._T_top = T_top
        self._P_surface = P_surface
        self._P_top = P_top
        self._smooth_window = smoothing_window
        self.generate_pressure_fitting_params()
        self.generate_temperature_fitting_params()

    @fitparam(param_name='T_surface',
              param_latex='$T_\\mathrm{surf}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureSurface(self):
        """Temperature at planet surface in Kelvin"""
        return self._T_surface

    @temperatureSurface.setter
    def temperatureSurface(self, value):
        self._T_surface = value

    @fitparam(param_name='T_top',
              param_latex='$T_\\mathrm{top}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureTop(self):
        """Temperature at top of atmosphere in Kelvin"""
        return self._T_top

    @temperatureTop.setter
    def temperatureTop(self, value):
        self._T_top = value

    @fitparam(param_name='P_surface',
              param_latex='$P_\\mathrm{surf}$',
              default_fit=False,
              default_bounds=[1e3, 1e2],
              default_mode='log')
    def pressureSurface(self):
        return self._P_surface

    @pressureSurface.setter
    def pressureSurface(self, value):
        self._P_surface = value

    @fitparam(param_name='P_top',
              param_latex='$P_\\mathrm{top}$',
              default_fit=False,
              default_bounds=[1e-5, 1e-4],
              default_mode='log')
    def pressureTop(self):
        return self._P_top

    @pressureTop.setter
    def pressureTop(self, value):
        self._P_top = value

    def generate_pressure_fitting_params(self):
        """Generates the fitting parameters for the pressure points
        These are given the name ``P_point(number)`` for example, if two extra
        pressure points are defined between the top and surface then the
        fitting parameters generated are ``P_point0`` and ``P_point1``
        """

        bounds = [1e5, 1e3]
        for idx, val in enumerate(self._p_points):
            point_num = idx+1
            param_name = 'P_point{}'.format(point_num)
            param_latex = '$P_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._p_points[idx]

            def write_point(self, value, idx=idx):
                self._p_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s', fget_point)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'log', default_fit, bounds)

    def generate_temperature_fitting_params(self):
        """Generates the fitting parameters for the temeprature points
        These are given the name ``T_point(number)`` for example, if two extra
        temeprature points are defined between the top and surface then the
        fitting parameters generated are ``T_point0`` and ``T_point1``
        """

        bounds = [300, 2500]
        for idx, val in enumerate(self._t_points):
            point_num = idx+1
            param_name = 'T_point{}'.format(point_num)
            param_latex = '$T_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._t_points[idx]

            def write_point(self, value, idx=idx):
                self._t_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s %s', fget_point, fget_point(self))
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'linear', default_fit, bounds)

    @property
    def profile(self):

        Tnodes = [self._T_surface, *self._t_points, self._T_top]

        Psurface = self._P_surface
        if Psurface is None or Psurface < 0:
            Psurface = self.pressure_profile[0]

        Ptop = self._P_top
        if Ptop is None or Ptop < 0:
            Ptop = self.pressure_profile[-1]

        Pnodes = [Psurface, *self._p_points, Ptop]

        smooth_window = self._smooth_window

        TP = np.interp((np.log(self.pressure_profile[::-1])),
                       np.log(Pnodes[::-1]), Tnodes[::-1])

        # smoothing T-P profile
        wsize = int(self.nlayers*(smooth_window / 100.0))
        if (wsize % 2 == 0):
            wsize += 1
        TP_smooth = movingaverage(TP, wsize)
        border = np.int((len(TP) - len(TP_smooth))/2)

        foo = TP[::-1]
        foo[border:-border] = TP_smooth[::-1]

        return foo

    def write(self, output):
        temperature = super().write(output)

        temperature.write_scalar('T_surface', self._T_surface)
        temperature.write_scalar('T_top', self._T_top)
        temperature.write_array('temperature_points', np.array(self._t_points))

        P_surface = self._P_surface
        P_top = self._P_top
        if not P_surface:
            P_surface = -1
        if not P_top:
            P_top = -1

        temperature.write_scalar('P_surface', P_surface)
        temperature.write_scalar('P_top', P_top)
        temperature.write_array('pressure_points', np.array(self._p_points))

        temperature.write_scalar('smoothing_window', self._smooth_window)

        return temperature

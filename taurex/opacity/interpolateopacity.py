import numpy as np
from .opacity import Opacity
from taurex.util.math import intepr_bilin, interp_exp_and_lin, \
    interp_lin_only, interp_exp_only


class InterpolatingOpacity(Opacity):
    """
    Provides interpolation methods

    """

    def __init__(self, name, interpolation_mode='linear'):
        super().__init__(name)
        self._interp_mode = interpolation_mode

    @property
    def pressureMax(self):
        return self.pressureGrid[-1]

    @property
    def pressureMin(self):
        return self.pressureGrid[0]

    @property
    def temperatureMax(self):
        return self.temperatureGrid[-1]

    @property
    def temperatureMin(self):
        return self.temperatureGrid[0]


    @property
    def xsecGrid(self):
        raise NotImplementedError

    @property
    def logPressure(self):
        return np.log10(self.pressureGrid)

    @property
    def pressureBounds(self):
        return self.logPressure.min(), self.logPressure.max()

    @property
    def temperatureBounds(self):
        return self.temperatureGrid.min(), self.temperatureGrid.max()

    def find_closest_index(self, T, P):
        from taurex.util.util import find_closest_pair
        # t_min = self.temperatureGrid.searchsorted(T, side='right')-1
        # t_min = max(0, t_min)
        # t_max = t_min+1
        # t_max = min(len(self.temperatureGrid)-1, t_max)

        # p_min = self.pressureGrid.searchsorted(P, side='right')-1
        # p_min = max(0, p_min)
        # p_max = p_min+1
        # p_max = min(len(self.pressureGrid)-1, p_max)

        t_min, t_max = find_closest_pair(self.temperatureGrid, T)
        p_min, p_max = find_closest_pair(self.logPressure, P)

        return t_min, t_max, p_min, p_max

    def set_interpolation_mode(self, interp_mode):
        self._interp_mode = interp_mode.strip()

    def interp_temp_only(self, T, t_idx_min, t_idx_max, P, filt):
        Tmax = self.temperatureGrid[t_idx_max]
        Tmin = self.temperatureGrid[t_idx_min]
        fx0 = self.xsecGrid[P, t_idx_min, filt]
        fx1 = self.xsecGrid[P, t_idx_max, filt]

        if self._interp_mode == 'linear':
            return interp_lin_only(fx0, fx1, T, Tmin, Tmax)
        elif self._interp_mode == 'exp':
            return interp_exp_only(fx0, fx1, T, Tmin, Tmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_pressure_only(self, P, p_idx_min, p_idx_max, T, filt):
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]
        fx0 = self.xsecGrid[p_idx_min, T, filt]
        fx1 = self.xsecGrid[p_idx_max, T, filt]

        return interp_lin_only(fx0, fx1, P, Pmin, Pmax)

    def interp_bilinear_grid(self, T, P, t_idx_min, t_idx_max, p_idx_min, 
                             p_idx_max, wngrid_filter=None):

        self.debug('Interpolating %s %s %s %s %s %s', T, P,
                   t_idx_min, t_idx_max, p_idx_min, p_idx_max)

        check_pressure_max = P >= self.pressureMax
        check_temperature_max = T >= self.temperatureMax

        min_pressure, max_pressure = self.pressureBounds
        min_temperature, max_temperature = self.temperatureBounds

        check_pressure_max = P >= max_pressure
        check_temperature_max = T >= max_temperature

        check_pressure_min = P < min_pressure
        check_temperature_min = T < min_temperature

        self.debug('Check pressure min/max %s/%s',
                   check_pressure_min, check_pressure_max)
        self.debug('Check temeprature min/max %s/%s',
                   check_temperature_min, check_temperature_max)
        # Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug('Maximum Temperature pressure reached. Using last')
            return self.xsecGrid[-1, -1, wngrid_filter].ravel()

        if check_pressure_min and check_temperature_min:
            return np.zeros_like(self.xsecGrid[0, 0, wngrid_filter]).ravel()

        # Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, -1, wngrid_filter)

        # Max temperature
        if check_temperature_max:
            self.debug('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, -1, wngrid_filter)



        if check_pressure_min:
            self.debug('Min pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, 0, wngrid_filter).ravel()

        if check_temperature_min:
            self.debug('Min temeprature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, 0, wngrid_filter).ravel()

        q_11 = self.xsecGrid[p_idx_min, t_idx_min][wngrid_filter].ravel()
        q_12 = self.xsecGrid[p_idx_min, t_idx_max][wngrid_filter].ravel()
        q_21 = self.xsecGrid[p_idx_max, t_idx_min][wngrid_filter].ravel()
        q_22 = self.xsecGrid[p_idx_max, t_idx_max][wngrid_filter].ravel()

        Tmax = self.temperatureGrid[t_idx_max]
        Tmin = self.temperatureGrid[t_idx_min]
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def compute_opacity(self, temperature, pressure, wngrid=None):
        import math
        logpressure = math.log10(pressure)
        return self.interp_bilinear_grid(temperature, logpressure, *self.find_closest_index(temperature, logpressure), wngrid) / 10000

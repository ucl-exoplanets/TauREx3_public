from taurex.log import Logger
import numpy as np
from .opacity import Opacity
from taurex.util.math import *


class InterpolatingOpacity(Opacity):
    """
    Provides interpolation methods

    """

    def __init__(self, name, interpolation_mode='linear'):
        super().__init__(name)
        self._interp_mode = interpolation_mode

    @property
    def xsecGrid(self):
        raise NotImplementedError

    def find_closest_index(self, T, P):
        t_min = self.temperatureGrid.searchsorted(T, side='right')-1
        t_max = t_min+1

        p_min = self.pressureGrid.searchsorted(P, side='right')-1
        p_max = p_min+1

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
        Pmax = self.pressureGrid[p_idx_max]
        Pmin = self.pressureGrid[p_idx_min]
        fx0 = self.xsecGrid[p_idx_min, T, filt]
        fx1 = self.xsecGrid[p_idx_max, T, filt]

        return interp_lin_only(fx0, fx1, P, Pmin, Pmax)

    def interp_bilinear_grid(self, T, P, t_idx_min, t_idx_max, p_idx_min, p_idx_max, wngrid_filter=None):

        self.debug('Interpolating %s %s %s %s %s %s', T, P,
                   t_idx_min, t_idx_max, p_idx_min, p_idx_max)

        if p_idx_max == 0 and t_idx_max == 0:

            return np.zeros_like(self.xsecGrid[0, 0, wngrid_filter]).ravel()

        check_pressure_max = P >= self._max_pressure
        check_temperature_max = T >= self._max_temperature

        check_pressure_min = P < self._min_pressure
        check_temperature_min = T < self._min_temperature

        self.debug('Check pressure min/max %s/%s',
                   check_pressure_min, check_pressure_max)
        self.debug('Check temeprature min/max %s/%s',
                   check_temperature_min, check_temperature_max)
        # Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug('Maximum Temperature pressure reached. Using last')
            return self.xsecGrid[-1, -1, wngrid_filter].ravel()

        # Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, -1, wngrid_filter)

        # Max temperature
        if check_temperature_max:
            self.debug('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, -1, wngrid_filter)

        if check_pressure_min and check_temperature_min:
            return self.xsecGrid[0, 0, wngrid_filter].ravel()

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
        Pmax = self.pressureGrid[p_idx_max]
        Pmin = self.pressureGrid[p_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def compute_opacity(self, temperature, pressure, wngrid=None):

        return self.interp_bilinear_grid(temperature, pressure, *self.find_closest_index(temperature, pressure), wngrid) / 10000

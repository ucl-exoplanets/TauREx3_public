from taurex.model import ForwardModel
import numpy as np

import pickle
import pylightcurve as plc
from .lightcurvedata import LightCurveData
from taurex.data.fittable import fitparam


class LightCurveModel(ForwardModel):
    """A base class for producing forward models"""

    @property
    def defaultBinner(self):
        from taurex.binning.lightcurvebinner import LightcurveBinner
        return LightcurveBinner

    def __init__(self, forward_model, file_loc, instruments=None):
        super().__init__('LightCurveModel')

        self._forward_model = forward_model
        self.file_loc = file_loc
        self._load_file()

        self._load_orbital_profile()

        if instruments is None:
            instruments = 'all'

        if instruments == 'all' or 'all' in instruments:
            instruments = LightCurveData.availableInstruments

        self._load_instruments(instruments)
        self._load_ldcoeff()
        self._initialize_lightcurves()
        self.setup_binner()

    def setup_binner(self):
        from taurex.data.spectrum import ArraySpectrum
        self._wngrid = 10000 / self.lc_data['obs_spectrum'][:, 0]
        self._arr_spec = ArraySpectrum(self.lc_data['obs_spectrum'])
        self._binner = self._arr_spec.create_binner()

    def initialize_profiles(self):
        self._forward_model.initialize_profiles()

    def _load_file(self):
        # input data from lightcurve, not from spectrum.

        with open(self.file_loc, 'rb') as f:
            self.lc_data = pickle.load(f, encoding='latin1')

    def _load_orbital_profile(self):
        """Load orbital information"""
        self.info('Load lc information')

        # new
        self.mid_time = self.lc_data['orbital_info']['mt']
        self._inclination = self.lc_data['orbital_info']['i']
        self.period = self.lc_data['orbital_info']['period']
        self.periastron = self.lc_data['orbital_info']['periastron']
        self.sma_over_rs = self.lc_data['orbital_info']['sma_over_rs']
        self.ecc = self.lc_data['orbital_info']['e']

    def _load_ldcoeff(self):
        # new

        self.ld_coeff_file = np.array([])
        for i in self._instruments:
            self.ld_coeff_file = np.append(
                self.ld_coeff_file,
                self.lc_data[i.instrumentName]['ld_coeff']).reshape(-1, 4)
        assert np.shape(self.ld_coeff_file)[
            1] == 4, "please use 4 ldcoeff law."

    def _load_instruments(self, instruments):
        self._instruments = []
        # new
        ins_keys = self.lc_data.keys()
        for ins in instruments:
            if ins in ins_keys:
                self.info('Loading {} light curves'.format(ins))
                self._instruments.append(
                    LightCurveData.fromInstrumentName(ins, self.lc_data))
            else:
                self.info('Could not find {} in instrument keys'.format(ins))
        new_instrument_list = sorted(
            self._instruments, key=lambda x: x.wavelengthRegion[0],
            reverse=True)
        self._instruments = new_instrument_list

    def _initialize_lightcurves(self):

        nFactor = [ins.minNFactors for ins in self._instruments]

        min_n_factors = None

        max_n_factors = None
        if len(nFactor) > 0:
            min_n_factors = np.concatenate(nFactor)

        nFactor = [ins.maxNFactors for ins in self._instruments]
        if len(nFactor) > 0:
            max_n_factors = np.concatenate(nFactor)

        if max_n_factors is not None and min_n_factors is not None:
            self.create_normalization_fitparams()

            for idx, value in enumerate(zip(min_n_factors, max_n_factors)):
                min_n, max_n = value
                self.modify_bounds('Nfactor_{}'.format(idx), [min_n, max_n])

        min_time = min([ins.timeSeries.min() for ins in self._instruments])

        max_time = max([ins.timeSeries.max() for ins in self._instruments])

        self.modify_bounds('mid_transit_time', [min_time, max_time])

    @fitparam(param_name='sma_over_rs', param_latex='sma_over_rs',
              default_mode='linear', default_fit=False,
              default_bounds=[1.001, 60.0])
    def semiMajorAxisOverRs(self):
        return self.sma_over_rs

    @semiMajorAxisOverRs.setter
    def semiMajorAxisOverRs(self, value):
        self.sma_over_rs = value

    @fitparam(param_name='mid_transit_time', param_latex='mid_transit_time',
              default_mode='linear', default_fit=False,
              default_bounds=[1.001, 2.0])
    def midTransitTime(self):
        return self.mid_time

    @midTransitTime.setter
    def midTransitTime(self, value):
        self.mid_time = value

    @fitparam(param_name='inclination', param_latex='inclination',
              default_mode='linear', default_fit=False,
              default_bounds=[40.0, 90.0])
    def inclination(self):
        return self._inclination

    @inclination.setter
    def inclination(self, value):
        self._inclination = value

    @property
    def temperatureProfile(self):
        return self._forward_model.temperatureProfile

    @property
    def pressureProfile(self):
        return self._forward_model.pressureProfile

    @property
    def densityProfile(self):
        return self._forward_model.densityProfile

    @property
    def scaleheight_profile(self):
        return self._forward_model.scaleheight_profile

    @property
    def chemistry(self):
        return self._forward_model.chemistry

    @property
    def gravity_profile(self):
        return self._forward_model.gravity_profile

    @property
    def pressure(self):
        return self._forward_model.pressure

    @property
    def altitudeProfile(self):
        return self._forward_model.altitudeProfile

    def create_normalization_fitparams(self):
        import itertools

        ins_name = list(itertools.chain(
            *tuple([[ins.instrumentName]*ins.minNFactors.shape[0]
                    for ins in self._instruments])))

        ins_number = list(itertools.chain(
            *tuple([list(range(ins.minNFactors.shape[0]))
                    for ins in self._instruments])))

        for idx, val in enumerate(zip(ins_name, ins_number)):
            name, no = val
            param_name = 'Nfactor_{}'.format(idx)
            param_latex = '{}_{}'.format(name, no)

            default_fit = False
            default_bounds = [0, 1]
            default_mode = 'linear'

            def readN(self, idx=idx):
                return self._nfactor[idx]

            def writeN(self, value, idx=idx):
                self._nfactor[idx] = value

            self.add_fittable_param(
                param_name, param_latex, readN, writeN, default_mode,
                default_fit, default_bounds)

    def instrument_light_curve(self, model, wlgrid):
        """Combine light-curves from different instrucments together."""
        result = np.array([])
        sma_over_rs_value = self.sma_over_rs
        inclination_value = self.inclination
        mid_time_value = self.mid_time
        try:
            Nfactor = self._nfactor
        except AttributeError:
            Nfactor = np.ones_like(wlgrid)

        result = []

        for ins in self._instruments:
            min_wl, max_wl = ins.wavelengthRegion
            index = (wlgrid > min_wl) & (wlgrid < max_wl)

            lc = self.light_curve_chain(model[index],
                                        time_array=ins.timeSeries,
                                        period=self.period,
                                        sma_over_rs=sma_over_rs_value,
                                        eccentricity=self.ecc,
                                        inclination=inclination_value,
                                        periastron=self.periastron,
                                        mid_time=mid_time_value,
                                        ldcoeff=self.ld_coeff_file[index],
                                        Nfactor=Nfactor[index])
            result.append(lc)

        return np.concatenate(result)

    def light_curve_chain(self, model, time_array, period, sma_over_rs,
                          eccentricity, inclination, periastron,
                          mid_time, ldcoeff, Nfactor):
        """Create model light-curve and lightcurve chain."""
        from taurex.model import TransmissionModel, EmissionModel
        result = []
        sqrt_model = np.sqrt(model)
        # self.info('Creating Lightcurve chain.')
        for n in range(len(sqrt_model)):
            if isinstance(self._forward_model, TransmissionModel):
                self.debug('Using Transit')
                transit_light_curve = plc.transit('claret', ldcoeff[n],
                                                  sqrt_model[n], period,
                                                  sma_over_rs, eccentricity,
                                                  inclination, periastron,
                                                  mid_time, time_array)
                result.append(transit_light_curve * Nfactor[n])
            elif isinstance(self._forward_model, EmissionModel):
                self.debug('Using Eclipse')
                rp_over_rs = (self._forward_model.planet.fullRadius /
                              self._forward_model.star.radius)
                self.debug('rp_over_rs %s', rp_over_rs)
                self.debug('fp_over_fs %s', sqrt_model)

                eclipse_light_curve = plc.eclipse(sqrt_model[n], rp_over_rs,
                                                  period,
                                                  sma_over_rs, eccentricity,
                                                  inclination, periastron,
                                                  mid_time, time_array)
                result.append(eclipse_light_curve * Nfactor[n])

        self.debug('Result %s', result)
        return np.concatenate(result)

    def build(self):
        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())

        self._forward_model.build()

        self._fitting_parameters.update(self._forward_model.fittingParameters)

    @property
    def nativeWavenumberGrid(self):
        return self._forward_model.nativeWavenumberGrid

    def model(self, wngrid=None, cutoff_grid=True):
        """Computes the forward model for a wngrid"""
        # new

        native_grid, model, tau, extra = self._forward_model.model(
            self._wngrid, cutoff_grid)
        binned_model = self._binner.bindown(native_grid, model)
        wlgrid = 10000/self._wngrid
        result = self.instrument_light_curve(binned_model[1], wlgrid)

        return self._wngrid, result, tau, \
            [native_grid, model, binned_model[1], extra]

    def compute_error(self, samples, wngrid=None, binner=None):
        from taurex.util.math import OnlineVariance
        tp_profiles = OnlineVariance()
        active_gases = OnlineVariance()
        inactive_gases = OnlineVariance()

        lc_spectrum = OnlineVariance()

        native_spectrum = OnlineVariance()
        binned_spectrum = OnlineVariance()

        for weight in samples():

            grid, lc, tau, extra = self.model(wngrid=wngrid, cutoff_grid=False)
            native_grid, native, binned, _ = extra

            tp_profiles.update(self.temperatureProfile, weight=weight)
            active_gases.update(
                self.chemistry.activeGasMixProfile, weight=weight)
            inactive_gases.update(
                self.chemistry.inactiveGasMixProfile, weight=weight)

            native_spectrum.update(native, weight=weight)
            binned_spectrum.update(binned, weight=weight)
            lc_spectrum.update(lc, weight=weight)

        profile_dict = {}
        spectrum_dict = {}

        tp_std = np.sqrt(tp_profiles.parallelVariance())
        active_std = np.sqrt(active_gases.parallelVariance())
        inactive_std = np.sqrt(inactive_gases.parallelVariance())

        profile_dict['temp_profile_std'] = tp_std
        profile_dict['active_mix_profile_std'] = active_std
        profile_dict['inactive_mix_profile_std'] = inactive_std

        spectrum_dict['native_std'] = np.sqrt(
            native_spectrum.parallelVariance())
        spectrum_dict['binned_std'] = np.sqrt(
            binned_spectrum.parallelVariance())
        spectrum_dict['lightcurve_std'] = np.sqrt(
            lc_spectrum.parallelVariance())
        return profile_dict, spectrum_dict

    def model_contrib(self, wngrid=None, cutoff_grid=True):
        # new
        wngrid = 10000 / self.lc_data['obs_spectrum'][:, 0]
        native_grid, contribs = self._forward_model.model_contrib(
            wngrid, cutoff_grid)
        binner = self._binner
        all_contrib_dict = {}

        wlgrid = 10000/wngrid

        for contrib_name, main_contribution in contribs.items():

            model, tau, extras = main_contribution
            binned = binner.bindown(native_grid, model)[1]

            result = self.instrument_light_curve(binned, wlgrid)
            all_contrib_dict[contrib_name] = (
                result, tau, [native_grid, model, binned, extras])

        return native_grid, all_contrib_dict

    def model_full_contrib(self, wngrid=None, cutoff_grid=True):
        """Computes the forward model for a wngrid for each contribution"""

        # new
        wngrid = 10000 / self.lc_data['obs_spectrum'][:, 0]

        native_grid, contrib_res = self._forward_model.model_full_contrib(
            wngrid, cutoff_grid)
        binner = self._binner

        self.info('Computing lightcurve contribution')
        wlgrid = 10000/wngrid
        lc_contrib_res = {}

        # Loop through each contribtuion
        for contrib_name, contrib_list in contrib_res.items():

            lc_contrib_list = []

            for name, native, tau, extra in contrib_list:

                binned = binner.bindown(native_grid, native)[1]
                result = self.instrument_light_curve(binned, wlgrid)

                new_packed = name, result, tau, (native_grid,
                                                 native, binned, extra)

                lc_contrib_list.append(new_packed)

            lc_contrib_res[contrib_name] = lc_contrib_list

        return native_grid, lc_contrib_res

    def write(self, output):
        lc = output.create_group('Lightcurve')

        lc_grps = lc.create_group('Instrument')
        for ins in self._instruments:
            ins.write(lc_grps)

        lc.write_scalar('mid_time', self.mid_time)
        lc.write_scalar('inclination', self._inclination)
        lc.write_scalar('period', self.period)
        lc.write_scalar('periastron', self.periastron)
        lc.write_scalar('sma_over_rs', self.sma_over_rs)
        lc.write_scalar('eccentricity', self.ecc)

        self._forward_model.write(output)

        return None

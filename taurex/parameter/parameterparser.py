import configobj
from taurex.log import Logger
from .factory import *

class ParameterParser(Logger):

    def __init__(self):
        super().__init__('ParamParser')
        self._read = False
    

    def transform(self, section, key):
        val = section[key]
        newval = val
        if isinstance(val, list):
            try:
                newval = list(map(float,val))

            except:
                pass
        elif isinstance(val, (str)):
            if val.lower() in ['true',  'yes', 'yeah', 'yup', 'certainly', 'uh-huh',]:
                newval = True
            elif val.lower() in ['false',  'no', 'nope', 'no-way', 'hell-no',]:
                newval = False
            else:
                try:
                    newval = float(val)
                except:
                    pass
        section[key]=newval
        return newval


    def setup_globals(self):
        from taurex.cache import CIACache,OpacityCache
        config = self._raw_config.dict()
        if 'Global' in config:
            try:
                OpacityCache().set_opacity_path(config['Global']['xsec_path'])
            except KeyError:
                self.warning('No xsec path set, opacities cannot be used in model')
            try:
                
                OpacityCache().set_interpolation(config['Global']['xsec_interpolation'])
                self.info('Interpolation mode set to {}'.format(config['Global']['xsec_interpolation']))
            except KeyError:
                self.info('Interpolation mode set to linear')

            try:
                CIACache().set_cia_path(config['Global']['cia_path'])
            except KeyError:
                self.warning('No cia path set, cia cannot be used in model')
            
            try:
                OpacityCache().set_memory_mode(config['Global']['xsec_in_memory'])
            except KeyError:
                self.warning('Xsecs will be loaded in memory')
            
            try:
                OpacityCache().enable_radis(config['Global']['use_radis'])
            except KeyError:
                self.warning('Radis is disabled')

            try:
                wn_start,wn_end,wn_points = config['Global']['radis_grid']

                OpacityCache().set_radis_wavenumber(wn_start,wn_end,wn_points)
            except KeyError:
                self.warning('Radis default grid will be used')


    def read(self,filename):
        import os.path
        if not os.path.isfile(filename):
            raise Exception('Input file {} does not exist'.format(filename))
        self._raw_config = configobj.ConfigObj(filename)
        self.debug('Raw Config file is {}, filename is {}'.format(self._raw_config,filename))
        self._raw_config.walk(self.transform)
        config = self._raw_config.dict()
        self.debug('Config file is {}, filename is {}'.format(config,filename))

    def generate_lightcurve(self):
        config = self._raw_config.dict()
        if 'Lightcurve' in config:
            from taurex.model.lightcurve.lightcurve import LightCurveModel
            model = self.generate_model()
            lightcurvefile = config['Lightcurve']['lc_pickle']
            return LightCurveModel(model,lightcurvefile)
        else:
            raise KeyError

    
    def generate_appropriate_model(self):

        try:
            return self.generate_lightcurve()
        except KeyError:
            return self.generate_model()

    def generate_instrument(self, binner=None):

        config = self._raw_config.dict()
        if 'Instrument' in config:
            inst_config = config['Instrument']

            try:
                num_obs = inst_config.pop('num_observations')
            except KeyError:
                num_obs = 1

            if 'instrument' in inst_config:
                if inst_config['instrument'].lower() in ('snr', 'signalnoise', ):
                    inst = self.create_snr(binner, inst_config)
                    return inst,num_obs
            inst = create_instrument(inst_config)
            return inst,num_obs
        else:
            return None

    def create_snr(self,binner,config):
        from taurex.instruments.snr import SNRInstrument
        if binner is None:
            self.critical('Binning must be defined for SNR instrument')
            raise ValueError('Binning must be defined for SNR instrument')
        else:
            SNR = 10
            if 'SNR' in config:
                SNR = config['SNR']
            
            return SNRInstrument(SNR=SNR,binner=binner)

    def generate_optimizer(self):
        config = self._raw_config.dict()
        if 'Optimizer' in config:
            return create_optimizer(config['Optimizer'])
        else:
            None

    def generate_observation(self):

        config = self._raw_config.dict()
        if 'Observation' in config:
            observation_config = config['Observation']
            if 'lightcurve' in observation_config:
                from taurex.data.spectrum.lightcurve import ObservedLightCurve
                return ObservedLightCurve(observation_config['lightcurve'])

            elif 'observed_spectrum' in observation_config:
                from taurex.data.spectrum.observed import ObservedSpectrum
                return ObservedSpectrum(observation_config['observed_spectrum'])
            elif 'phasecurve_path' in observation_config:
                self.info('Phase curves to be implemented soon...... :D ')
            elif 'taurex_spectrum' in observation_config:
                if observation_config['taurex_spectrum'] == 'self':
                    return 'self'
                from taurex.data.spectrum.taurex import TaurexSpectrum
                return TaurexSpectrum(observation_config['taurex_spectrum'])
            elif 'iraclis_spectrum' in observation_config:
                from taurex.data.spectrum.iraclis import IraclisSpectrum
                return IraclisSpectrum(observation_config['iraclis_spectrum'])
            else:
                self.warning('No observation specified........')
                return None
        return None
    
    def create_manual_binning(self, config):
        import numpy as np
        import math
        from taurex.binning import FluxBinner, SimpleBinner
        
        binning_class = SimpleBinner

        if 'accurate' in config:
            if config['accurate']:
                binning_class = FluxBinner
        
        # Handle wavelength grid
        wngrid = None
        if 'wavelength_grid' in config:
            start, end, size = config['wavelength_grid']
            wngrid = 10000/np.linspace(start, end, int(size))
            wngrid = np.sort(wngrid)

        elif 'wavenumber_grid' in config:
            start, end, size = config['wavenumber_grid']
            wngrid = np.linspace(start, end, int(size))

        elif 'log_wavenumber_grid' in config:
            start, end, size = config['log_wavenumber_grid']
            start = math.log10(start)
            end = math.log10(end)
            wngrid = np.logspace(start, end, int(size))

        elif 'log_wavelength_grid' in config:
            start, end, size = config['log_wavelength_grid']
            start = math.log10(start)
            end = math.log10(end)
            wngrid = np.sort(10000/np.logspace(start, end, int(size)))
        elif 'wavelength_res' in config:
            from taurex.util.util import create_grid_res
            start, end, res = config['wavelength_res']
            wlgrid = create_grid_res(res, start, end)[:, 0].flatten()
            wngrid = 10000/wlgrid[::-1]

        if wngrid is None:
            self._logger.error('manual was selected and no grid was given.'
                               '(Use wavelength_grid, wavenumber_grid or log versions)')
            raise Exception('manual selected but no grid given')

        return binning_class(wngrid), wngrid

    def generate_binning(self):

        config = self._raw_config.dict()
        if 'Binning' in config:
            binning_config = config['Binning']
            if 'bin_type' in binning_config:
                bin_type = binning_config['bin_type'].lower()

                if bin_type == 'native':
                    return 'native'
                elif bin_type == 'observed':
                    return 'observed'
                elif bin_type == 'manual':
                    return self.create_manual_binning(binning_config)
            else:
                return None
        return None
    # def generate_spectrum(self):
    #     import numpy as np

    #     config = self._raw_config.dict()
    #     observed = None
    #     if 'Spectrum' in config:
    #         spectrum_config = config['Spectrum']
    #         if 'lightcurve' in spectrum_config:
    #             from taurex.data.spectrum.lightcurve import ObservedLightCurve
    #             observed = ObservedLightCurve(spectrum_config['lightcurve'])

    #         elif 'observed_spectrum' in spectrum_config:
    #             from taurex.data.spectrum.observed import ObservedSpectrum
    #             observed = ObservedSpectrum(spectrum_config['observed_spectrum'])

    #         if 'grid_type' in spectrum_config:
    #             grid_type = spectrum_config['grid_type']

    #             if grid_type == 'observed':
    #                 if observed is not None:
    #                     return observed,observed.wavenumberGrid
    #                 else:
    #                     self.critical('grid type is observed yet no observed_spectrum is defined!!!')
    #                     raise Exception('No observed spectrum defined for observed grid_type')
    #             elif grid_type == 'native':
    #                 return observed,None
    #             elif grid_type == 'manual':

    #                 if 'wavenumber_grid' in spectrum_config:
    #                     start,end,size = spectrum_config['wavenumber_grid']
    #                     return observed,np.linspace(start,end,int(size))
    #                 elif 'wavelength_grid' in spectrum_config:
    #                     start,end,size = spectrum_config['wavelength_grid']
    #                     return observed,np.linspace(10000/end,10000/start,int(size))
    #                 else:
    #                    self.critical('grid type is manual yet neither wavelength_grid or wavenumber_grid is defined')
    #                    raise Exception('wavenumber_grid/wavelength_grid not defined in input for manual grid_type')
    #             else:
    #                 return observed,None

    def generate_model(self, chemistry=None, pressure=None,
                       temperature=None, planet=None,
                       star=None):
        config = self._raw_config.dict()
        if 'Model' in config:
            if chemistry is None:
                chemistry = self.generate_chemistry_profile()
            if pressure is None:
                pressure = self.generate_pressure_profile()
            if temperature is None:
                temperature = self.generate_temperature_profile()
            if planet is None:
                planet = self.generate_planet()
            if star is None:
                star = self.generate_star()
            model= create_model(config['Model'],chemistry,temperature,pressure,planet,star)
        else:
            return None
        
        return model
    def generate_chemistry_profile(self):
        config = self._raw_config.dict()
        if 'Chemistry' in config:
            return create_chemistry(config['Chemistry'])
        else:
            return None

    def generate_pressure_profile(self):
        config = self._raw_config.dict()
        if 'Pressure' in config:
            return create_pressure_profile(config['Pressure'])
        else:
            return None
    
    def generate_temperature_profile(self):
        config = self._raw_config.dict()
        if 'Temperature' in config:
            return create_temperature_profile(config['Temperature'])
        else:
            return None
    
    def generate_planet(self):
        config = self._raw_config.dict()

        if 'Planet' in config:
            return create_planet(config['Planet'])
        else:
            return None
    def generate_star(self):
        config = self._raw_config.dict()

        if 'Star' in config:
            return create_star(config['Star'])
        else:
            return None


    def generate_fitting_parameters(self):
        config = self._raw_config.dict()
        if 'Fitting' in config:
            fitting_config = config['Fitting']

            fitting_params = {}

            for key,value in fitting_config.items():
                fit_param,fit_type=key.split(':')
                if not fit_param in fitting_params:
                    fitting_params[fit_param] = {'fit':None,'bounds':None,'mode':None,'factor':None}
                fitting_params[fit_param][fit_type]=value

        return fitting_params





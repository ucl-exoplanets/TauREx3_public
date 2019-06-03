from taurex.model import ForwardModel
import numpy as np
import math
import pathlib
import pickle
import pylightcurve as plc
from .lightcurvedata import LightCurveData
from taurex.data.fittable import fitparam

class LightCurveModel(ForwardModel):
    """A base class for producing forward models"""




    def __init__(self,forward_model,file_loc,instruments=None):
        super().__init__('LightCurveModel')

        self._forward_model = forward_model
        self.file_loc = file_loc
        self._load_file()
        self._load_ldcoeff()
        self._load_orbital_profile()

        if instruments is None:
            instruments = 'all'
        
        if instruments == 'all' or 'all' in instruments:
            instruments = ['wfc3','spitzer','stis']

        self._load_instruments(instruments)

        self._initialize_lightcurves()


            

    def _load_file(self):
        # input data from lightcurve, not from spectrum.

        with open(self.file_loc,'rb') as f:
            self.lc_data = pickle.load(f,encoding='latin1')

    def _load_orbital_profile(self):
        """Load orbital information"""
        self.info('Load lc information')
        self.mid_time = self.lc_data['corr_orbital'][0]
        self._inclination = self.lc_data['corr_orbital'][1]
        self.period = self.lc_data['corr_orbital'][2]
        self.periastron = self.lc_data['corr_orbital'][3]
        self.sma_over_rs = self.lc_data['corr_orbital'][4]
        self.ecc = self.lc_data['corr_orbital'][5]

    def _load_ldcoeff(self):
        self.ld_coeff_file = self.lc_data['ld_coeff']
        assert np.shape(self.ld_coeff_file)[1] == 4, "please use 4 ldcoeff law."


    def _load_instruments(self,instruments):
        self._instruments = []

        ins_keys = self.lc_data['data'].keys()

        for ins in instruments:
            if ins in ins_keys:
                self.info('Loading {} light curves'.format(ins))
                self._instruments.append(LightCurveData.fromInstrumentName(ins,self.lc_data))
            else:
                self.info('Could not find {} in instrument keys'.format(ins))

    # def _load_data_file(self,instruments):
    #     """load data from different instruments."""

    #     raw_data = []
    #     data_std = []

    #     for i in self.lc_data['data']:
    #         # raw data includes data and datastd.
    #         raw_data.append(self.lc_data['data'][i][:len(self.lc_data['data'][i])//2])
    #         data_std.append(self.lc_data['data'][i][len(self.lc_data['data'][i])//2:])
    
    def _initialize_lightcurves(self):

        min_n_factors = np.concatenate([ins.minNfactors for ins in self._instruments])
        max_n_factors = np.concatenate([ins.maxNfactors for ins in self._instruments]) 
        
        self._nfactor = np.ones_like(min_n_factors)

        self.create_normalization_fitparams()

        for idx,value in enumerate(zip(min_n_factors,max_n_factors)):
            min_n,max_n = value
            self.modify_bounds('Nfactor_{}'.format(idx),[min_n,max_n])


    @fitparam(param_name='sma_over_rs',param_latex='sma_over_rs',
            default_mode='linear',default_fit=False,default_bounds=[1.001,60.0])
    def semiMajorAxisOverRs(self):
        return self.sma_over_rs
    
    @semiMajorAxisOverRs.setter
    def semiMajorAxisOverRs(self,value):
        self.sma_over_rs = value

    @fitparam(param_name='mid_transit_time',param_latex='mid_transit_time',
            default_mode='linear',default_fit=False,default_bounds=[1.001,2.0])
    def midTransitTime(self):
        return self.mid_time
    
    @midTransitTime.setter
    def midTransitTime(self,value):
        self.mid_time = value

    @fitparam(param_name='inclination',param_latex='inclination',
            default_mode='linear',default_fit=False,default_bounds=[40.0,90.0])
    def inclination(self):
        return self._inclination
    
    @inclination.setter
    def inclination(self,value):
        self._inclination = value


    def create_normalization_fitparams(self):
        import itertools
        

        ins_name = itertools.chain(*tuple([[ins.instrumentName]*ins.minNfactors.shape[0] for ins in self._instruments]))
        ins_number = itertools.chain(*tuple([list(range(ins.minNfactors.shape[0])) for ins in self._instruments]))

        for idx,val in enumerate(zip(ins_name,ins_number)):
            name,no = val
            param_name = 'Nfactor_{}'.format(idx)
            param_latex = '{}_{}'.format(name,no)

            default_fit = False
            default_bounds = [0,1]
            default_mode = 'linear'

            def readN(self,idx=idx):
                return self._nfactor[idx]
            def writeN(self,value,idx=idx):
                self._nfactor[idx] = value

            self.add_fittable_param(param_name,param_latex,readN,writeN,default_mode,default_fit,default_bounds)    

    def instrument_light_curve(self,model,wlgrid):
        """Combine light-curves from different instrucments together."""
        result = np.array([])
        sma_over_rs_value = self.sma_over_rs
        inclination_value = self.inclination
        mid_time_value = self.mid_time
        Nfactor = self._nfactor

        result = []
        for ins in self._instruments:
            min_wl,max_wl = ins.wavelengthRegion
            index = (wlgrid > min_wl) & (wlgrid < max_wl)
            lc = self.light_curve_chain(model[index], time_array=ins.timeArray, period=self.period,
                                                        sma_over_rs=sma_over_rs_value, eccentricity=self.ecc,
                                                        inclination=inclination_value, periastron=self.periastron,
                                                        mid_time=mid_time_value, ldcoeff=self.ld_coeff_file[index],
                                                        Nfactor=Nfactor[index])
            result.append(lc)

        return np.concatenate(result)
        #!# incomplete
    
    def light_curve_chain(self, model, time_array, period, sma_over_rs, eccentricity, inclination, periastron,
                          mid_time, ldcoeff, Nfactor ):
        """Create model light-curve and lightcurve chain."""
        result = []
        # self.info('Creating Lightcurve chain.')
        for n in range(len(model)):
            transit_light_curve = plc.transit('claret', ldcoeff[n], np.sqrt(model[n]), period,
                                          sma_over_rs, eccentricity, inclination, periastron,
                                          mid_time, time_array)
            result.append(transit_light_curve * Nfactor[n])

        return np.concatenate(result)


             
    
    def build(self):
        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())

        self._forward_model.build()
        
        self._fitting_parameters.update(self._forward_model.fittingParameters)

    @property
    def nativeWavenumberGrid(self):
        return self._forward_model.nativeWavenumberGrid


    def model(self,wngrid=None,return_contrib=False):
        """Computes the forward model for a wngrid"""
        binned_model,model,tau,contrib = self._forward_model.model(wngrid,return_contrib)
        if wngrid is None:
            wngrid = self.nativeWavenumberGrid
        
        wlgrid = 10000/wngrid

        result = self.instrument_light_curve(binned_model,wlgrid)


        return result,model,tau,contrib

    
    

    def write(self,output):
        raise NotImplementedError


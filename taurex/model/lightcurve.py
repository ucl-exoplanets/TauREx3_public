from .model import ForwardModel
import numpy as np
import math
import pathlib
import pickle
import pylightcurve as plc

from taurex.data.fittable import fitparam

class LightCurveModel(ForwardModel):
    """A base class for producing forward models"""




    def __init__(self,forward_model,file_loc,load_spitzer,load_wfc3,load_stis,load_nfactor,load_orbital):
        super().__init__('LightCurveModel')

        self._forward_model = forward_model
        self.file_loc = file_loc
        self.spitzer = load_spitzer
        self.wfc3 = load_wfc3
        self.stis = load_stis
        self.nfactor = load_nfactor
        self.orbital  = load_orbital
        self._load_file()
        self._load_ldcoeff()
        self._load_orbital_profile()
        self.load_wl_profile()
        self.load_data_file()


            

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
        self.seq_length = len(self.lc_data['lc_info'].T[0])
        self.orbital_list = [self.mid_time,self._inclination,self.period,self.periastron,self.sma_over_rs,self.ecc]

    def _load_ldcoeff(self):
        self.ld_coeff_file = self.lc_data['ld_coeff']
        assert np.shape(self.ld_coeff_file)[1] == 4, "please use 4 ldcoeff law."

    def load_data_file(self):
        """load data from different instruments."""
        Nfactor_range_list = []
        raw_data_list = np.array([])
        data_std_list = np.array([])

        for i in self.lc_data['data']:
            # raw data includes data and datastd.
            raw_data = self.lc_data['data'][i][:len(self.lc_data['data'][i])//2]
            data_std = self.lc_data['data'][i][len(self.lc_data['data'][i])//2:]



        if self.spitzer:
            instr = 'spitzer'
            self.time_series_spitzer = self.lc_data['time_series']['spitzer']
            raw_data = self.lc_data['data'][instr][:len(self.lc_data['data'][instr]) //2]
            data_std = self.lc_data['data'][instr][len(self.lc_data['data'][instr]) //2:]
            raw_data_list = np.append(raw_data_list,raw_data.flatten())
            data_std_list = np.append(data_std_list,data_std.flatten())
            max = np.max(raw_data, axis=1)
            min = np.min(raw_data, axis=1)
            Nfactor_range = np.column_stack([max, min])
            Nfactor_range_list.append(Nfactor_range)
        if self.wfc3:
            instr = 'wfc3'
            self.time_series_wfc3 = self.lc_data['time_series']['wfc3']
            raw_data = self.lc_data['data'][instr][:len(self.lc_data['data'][instr]) //2]
            data_std = self.lc_data['data'][instr][len(self.lc_data['data'][instr]) //2:]
            raw_data_list = np.append(raw_data_list, raw_data.flatten())
            data_std_list = np.append(data_std_list, data_std.flatten())
            max_n = np.max(raw_data, axis=1)
            min_n = np.min(raw_data, axis=1)
            Nfactor_range = np.column_stack([max_n, min_n])
            Nfactor_range_list.append((min_n,max_n))
            min_t = self.time_series_wfc3.min()
            max_t = self.time_series_wfc3.max()
            self.modify_bounds('mid_transit_time',[min_t,max_t])
        if self.stis:
            instr = 'stis'
            self.time_series_stis = self.lc_data['time_series']['stis']
            raw_data = self.lc_data['data'][instr][:len(self.lc_data['data'][instr]) //2]
            data_std = self.lc_data['data'][instr][len(self.lc_data['data'][instr]) //2:]
            raw_data_list = np.append(raw_data_list, raw_data.flatten())
            data_std_list = np.append(data_std_list, data_std.flatten())
            max = np.max(raw_data, axis=1)
            min = np.min(raw_data, axis=1)
            Nfactor_range = np.column_stack([max, min])
            Nfactor_range_list.append(Nfactor_range)
        

        for m,n in Nfactor_range_list:
            for idx,v in enumerate(zip(m,n)):
                
                self.modify_bounds('Nfactor_{}'.format(idx),list(v))
                #print(self._param_dict['Nfactor_{}'.format(idx)])
        return np.array(raw_data_list),np.array(data_std_list),np.array(Nfactor_range_list),np.array(self.time_series_wfc3)


    def load_wl_profile(self):
        """Load wavelength profile from the light-curve data, not from obs spectrum."""
        self.info('Create Observe Spectrum from Pipeline data')
        obs_spectrum = np.empty([len(self.lc_data['lc_info'][:, 0]), 4])
        obs_spectrum[:, 0] = self.lc_data['lc_info'][:, 0]
        obs_spectrum[:, 1] = self.lc_data['lc_info'][:, 3]
        obs_spectrum[:, 2] = self.lc_data['lc_info'][:, 1]
        obs_spectrum[:, 3] = self.lc_data['lc_info'][:, 2]
        # obs_spectrum = obs_spectrum[obs_spectrum[:, 0].argsort(axis=0)[::-1]]  # sort in wavenumber

        # assert obs_spectrum[:, 0].argmin() != 0, "list not sorted"
        self.obs_wlgrid = self.lc_data['lc_info'][:, 0]
        obs_wngrid = 10000. / self.obs_wlgrid
        obs_nwlgrid = len(self.obs_wlgrid)
        
        
        self._nfactor = np.ones_like(self.obs_wlgrid)
        self.create_normalization_fitparams()

        return obs_spectrum,self.obs_wlgrid,obs_wngrid,obs_nwlgrid

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

        for idx,_ in enumerate(self._nfactor):
            param_name = 'Nfactor_{}'.format(idx)
            param_latex = 'N_{}'.format(idx)

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
        if self.spitzer:
            index = np.logical_and(wlgrid > 3.4, wlgrid < 8.2)
            result = np.append(result,self.light_curve_chain(model[index], time_array=self.time_series_spitzer, period=self.period,
                                            sma_over_rs=sma_over_rs_value, eccentricity=self.ecc,
                                            inclination=inclination_value, periastron=self.periastron,
                                            mid_time=mid_time_value, ldcoeff=self.ld_coeff_file[index],
                                            Nfactor=Nfactor[index]))

        if self.wfc3:
            index = np.logical_and(wlgrid>1.1 ,wlgrid < 1.8)
            result = np.append(result,self.light_curve_chain(model[index], time_array=self.time_series_wfc3, period=self.period,
                           sma_over_rs=sma_over_rs_value, eccentricity=self.ecc,
                           inclination=inclination_value, periastron=self.periastron,
                           mid_time=mid_time_value, ldcoeff=self.ld_coeff_file[index],
                           Nfactor=Nfactor[index]))

        if self.stis:
            index = np.logical_and(wlgrid > 0.3, wlgrid < 1.0)
            result = np.append(result,self.light_curve_chain(model[index], time_array=self.time_series_stis, period=self.period,
                                            sma_over_rs=sma_over_rs_value, eccentricity=self.ecc,
                                            inclination=inclination_value, periastron=self.periastron,
                                            mid_time=mid_time_value, ldcoeff=self.ld_coeff_file[index],
                                            Nfactor=Nfactor[index]))

        return result
        #!# incomplete
    
    def light_curve_chain(self, model, time_array, period, sma_over_rs, eccentricity, inclination, periastron,
                          mid_time, ldcoeff, Nfactor ):
        """Create model light-curve and lightcurve chain."""
        result = np.array([])
        # self.info('Creating Lightcurve chain.')
        for n in range(len(model)):
            transit_light_curve = plc.transit('claret', ldcoeff[n], np.sqrt(model[n]), period,
                                          sma_over_rs, eccentricity, inclination, periastron,
                                          mid_time, time_array)
            result = np.append(result, transit_light_curve * Nfactor[n])

        return result

    def instrument_info(self,instr):
        """extract data, ldcoeff etc instrument specific data from the pickle file"""
        data = self.lc_data['data'][instr][:len(self.lc_data['data'][instr]) / 2]
        time_array = self.lc_data['time_series'][instr]
        return data, time_array, self.ld_coeff_file, self.obs_wlgrid


    def return_orbital(self):
        return self.orbital_list

             
    
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


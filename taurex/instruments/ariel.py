from taurex.instruments.instrument import Instrument
import astropy.units as u
try:
    import ArielRad as AR
    from ArielRad.target_list import BaseTargetList
except ImportError:
    self.error('ArielRad is not installed!!!')
    raise ImportError

import numpy as np

class PythonTargetList(BaseTargetList):


    

    def __init__(self, planet_keys, planet_data, star_keys, star_data,
                 planet_units=[u.day, u.K, u.m, u.R_earth, 1, u.M_earth,
                               u.g/u.mol, u.s, 1, 1, 1],
                 star_units=[u.M_sun, u.K, u.R_sun, u.pc, 1]):
        self._pk = planet_keys
        self._pd = planet_data
        self._sk = star_keys
        self._sd = star_data
        self._pu = planet_units
        self._su = star_units

        super().__init__()

    def read_data(self):
        pass
    
    def star_keys(self):
        return self._sk
    
    
    def star_data(self):
        return [[self._sd[0],*[a*b for a,b in zip(self._sd[1:],self._su)]]]
    
    def planet_keys(self):
        return self._pk
    
    def planet_data(self):
        return [[self._pd[0],*[a*b for a,b in zip(self._pd[1:],self._pu)]]]

class ArielInstrument(Instrument):


    def __init__(self, config_file=None,tier=2):
        super().__init__()

        try:
           
            import ArielRad as AR
            from ArielRad.targetlist_evaluate import compute_table
        except ImportError:
            self.error('ArielRad is not installed!!!')
            quit()
        
        if config_file is None:
            self.error('No config file given for ariel rad')
            raise ValueError('No config file given')
        
        self.tier = int(tier)

        self._opt = AR.Options(config_file).getopt()
        self.run, self.payload = self.get_configs()
        



    def get_configs(self):
        import ArielRad as AR
        """

        Returns
        -------
        option-class
            parsed run configuration
        option-class
            parsed payload configuration

        """

        opt = self._get_mase_mat_keys()
        payloadConfig = AR.Options(opt.configurationFile()).getopt()
        return opt, payloadConfig


    def _get_mase_mat_keys(self):
        import ArielRad as AR
        """
        Edits the input option from ArielRad to set the output keys

        Returns
        -------
        option-class object
            full runConfig option class with mase and mat keys set

        """
        set_out_keys = AR.options.Item()
        set_out_keys_values = 'name Wavelength BandWidth LeftBinEdge RightBinEdge NoiseOnTransitFloorStack'
        setattr(set_out_keys, '__val__', set_out_keys_values)
        opt = self._opt
        opt.maseOutputKeys = set_out_keys
        opt.matOutputKeys = set_out_keys
        return opt 


    def model_noise(self, model, model_res=None, num_observations=1):
        from taurex.binning import FluxBinner




        star = model.star

        star_keys = ["name","M", "Teff", "R", "D", "magk"]
        star_units=[u.M_sun, u.K, u.R_sun, u.pc, 1]
        star_data = ['TAUREX3.0',star.mass,star.temperature, star.radius,star.distance, star.magnitudeK]

        planet = model.planet

        planet_keys = ["name","P", "Teff", "a", "R", "albedo", "M", "MMW", "T14", "b",
                       "Gamma", "Nobs", "tier"]

        planet_data = ["WHATSUP",planet.orbitalPeriod, np.mean(model.temperatureProfile),
                       planet.distance, planet.radius, planet.albedo,
                       planet.mass, np.mean(model.chemistry.muProfile),
                       planet.transitTime, planet.impactParameter,
                       1, num_observations, self.tier]
        planet_units=[u.day, u.K, u.m, u.R_jupiter, 1, u.M_jupiter,
                               u.kg/u.mol, u.s, 1, 1, 1]


        ptl = PythonTargetList(planet_keys,planet_data,star_keys,star_data,planet_units=planet_units,star_units=star_units)

        ptl.name = 'TAUREX 3.0 BABY'


        te_outDict = AR.targetlist_evaluate.compute_table(self.payload, self.run, ptl, saveFile=False)
        _SNR, tsc_outDict, _fn = AR.targetlist_snr_calculator.run(self.run, inputData =te_outDict, fullOut = False, slimOut = False)
        dataDict = AR.make_ariel_tiers.run(self.run, inputData = tsc_outDict, fullOut = False, slimOut=False)

        if model_res is None:
            model_res = model.model()

        tier_name = 'tier{}'.format(self.tier)

        wavenumber = 10000/dataDict[tier_name]['TAUREX3.0']['Wavelength'].data
        sorted_id = wavenumber.argsort()

        wavenumber = wavenumber[sorted_id]

        leftbinEdge = 10000/dataDict[tier_name]['TAUREX3.0']['LeftBinEdge'].data[sorted_id]
        rightbinEdge = 10000/dataDict[tier_name]['TAUREX3.0']['RightBinEdge'].data[sorted_id]
        noise = dataDict[tier_name]['TAUREX3.0']['NoiseOnTransitFloorStack'].data[sorted_id]

        wn_width = np.abs(rightbinEdge-leftbinEdge)

        fb = FluxBinner(wavenumber,wn_width)

        wngrid, spectrum, error, grid_width = fb.bin_model(model_res)


        return wngrid, spectrum, noise,wn_width
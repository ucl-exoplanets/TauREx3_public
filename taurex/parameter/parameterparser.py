import configobj
from taurex.log import Logger
from .factory import create_gas_profile,create_pressure_profile,create_temperature_profile,create_klass,create_model

class ParameterParser(Logger):

    def __init__(self):
        super().__init__('ParamParser')
        self._read = False
    

    def transform(self,section,key):
        val = section[key]
        newval = val
        if isinstance(val,list):
            try:
                newval = list(map(float,val))

            except:
                pass
        elif isinstance(val, (str)):
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
                CIACache().set_cia_path(config['Global']['cia_path'])
            except KeyError:
                self.warning('No cia path set, cia cannot be used in model')



    def read(self,filename):
        import os.path
        if not os.path.isfile(filename):
            raise Exception('Input file {} does not exist'.format(filename))
        self._raw_config = configobj.ConfigObj(filename)
        self.debug('Raw Config file is {}, filename is {}'.format(self._raw_config,filename))
        self._raw_config.walk(self.transform)
        config = self._raw_config.dict()
        self.debug('Config file is {}, filename is {}'.format(config,filename))

    def generate_model(self):
        config = self._raw_config.dict()
        if 'Model' in config:
            gas = self.generate_gas_profile()
            pressure = self.generate_pressure_profile()
            temperature = self.generate_temperature_profile()
            planet = self.generate_planet()
            star = self.generate_star()
            model= create_model(config['Model'],gas,temperature,pressure,planet,star)
        else:
            raise Exception('No model header defined in input file')
        
        return model
    def generate_gas_profile(self):
        config = self._raw_config.dict()
        if 'Gas' in config:
            return create_gas_profile(config['Gas'])
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
            from taurex.data.planet import Planet
            return create_klass(config['Planet'],Planet)
        else:
            return None
    def generate_star(self):
        config = self._raw_config.dict()
        if 'Star' in config:
            from taurex.data.stellar.star import Star
            return create_klass(config['Star'],Star)



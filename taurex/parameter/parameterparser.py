import configobj
from .factory import create_gas_profile,create_pressure_profile,create_temperature_profile,create_klass

class ParameterParser(object):

    def __init__(self):
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

    def read(self,filename):
        self._config = configobj.ConfigObj(filename)
        self._config.walk(self.transform)

    def generate_model(self):
        pass
    
    def generate_gas_profile(self):
        if 'Gas' in self._config:
            return create_gas_profile(self._config['Gas'])
        else:
            return None

    def generate_pressure_profile(self):
        if 'Pressure' in self._config:
            return create_pressure_profile(self._config['Pressure'])
        else:
            return None
    
    def generate_temperature_profile(self):
        if 'Temperature' in self._config:
            return create_temperature_profile(self._config['Temeperature'])
        else:
            return None
    
    def generate_planet(self):
        if 'Planet' in self._config:
            from taurex.data.planet import Planet
            return create_klass(self._config['Planet'],Planet)
        else:
            return None
    def generate_star(self):
        if 'Star' in self._config:
            from taurex.data.stellar.star import Star
            return create_klass(self._config['Star'],Star)


    
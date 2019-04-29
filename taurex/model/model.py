from taurex.log import Logger
import numpy as np
import math

class ForwardModel(Logger):
    """A base class for producing forward models"""

    def __init__(self,name,opacities=None,
                            cia=None,
                            opacity_path=None,
                            cia_path=None,):
        super().__init__(name)
        self.opacity_dict = {}
        self.cia_dict = {}
        self.load_opacities(opacities,opacity_path)
        self.load_cia(cia,cia_path)    

    def add_opacity(self,opacity):
        self.info('Loading opacity {} into model'.format(opacity.moleculeName))
        if opacity.moleculeName in self.opacity_dict:
            self.error('Opacity with name {} already in opactiy dictionary {}'.format(opacity.moleculeName,self.opacity_dict.keys()))
            raise Exception('Opacity for molecule {} already exists')
        self.opacity_dict[opacity.moleculeName] = opacity            
    
    def load_opacity_from_path(self,path):
        from glob import glob
        import os
        from taurex.opacity import PickleOpacity
        glob_path = os.path.join(path,'*.pickle')

        file_list = glob(glob_path)
        self.debug('File list {}'.format(file_list))
        for files in file_list:
            op = PickleOpacity(files)
            self.add_opacity(op)



    def load_opacities(self,opacities,opacity_path):
        from taurex.opacity import Opacity
        if opacities is not None:
            if isinstance(opacities,(list,)):
                self.debug('Opacity passed is list')
                for opacity in opacities:
                    self.add_opacity(opacity)
            elif isinstance(opacities,Opacity):
                self.add_opacity(opacities)
            else:
                self.error('Unknown type {} passed into opacities, should be a list, single \
                     opacity or None if reading a path'.format(type(opacities)))
                raise Exception('Unknown type passed into opacities')
        elif opacity_path is not None:

            if isinstance(opacity_path,str):
                self.load_opacity_from_path(opacity_path)
            elif isinstance(opacity_path,(list,)):
                for path in opacity_path:
                    self.load_opacity_from_path(path)
            


    def add_cia(self,cia):
        self.info('Loading cia {} into model'.format(cia.pairName))
        if cia.pairName in self.cia_dict:
            self.error('cia with name {} already in opactiy dictionary {}'.format(cia.pairName,self.cia_dict.keys()))
            raise Exception('cia for molecule {} already exists')
        self.cia_dict[cia.pairName] = cia            
    
    def load_cia_from_path(self,path):
        from glob import glob
        from pathlib import Path
        import os
        from taurex.cia import PickleCIA
        glob_path = os.path.join(path,'*.db')

        file_list = glob(glob_path)
        self.debug('File list {}'.format(file_list))
        for files in file_list:
            pairname=Path(files).stem
            op = PickleCIA(files,pairname)
            self.add_cia(op)

    def load_cia(self,cias,cia_path):
        from taurex.cia import CIA
        if cias is not None:
            if isinstance(cias,(list,)):
                self.debug('cia passed is list')
                for cia in cias:
                    self.add_cia(cia)
            elif isinstance(cias,CIA):
                self.add_cia(cia)
            else:
                self.error('Unknown type {} passed into cia, should be a list, single \
                     cia or None if reading a path'.format(type(cia)))
                raise Exception('Unknown type passed into cia')
        elif cia_path is not None:

            if isinstance(cia_path,str):
                self.load_cia_from_path(cia_path)
            elif isinstance(cia_path,(list,)):
                for path in cia_path:
                    self.load_cia_from_path(path)        
    


    def model(self,wngrid):
        """Computes the forward model for a wngrid"""
        raise NotImplementedError

    


class SimpleForwardModel(ForwardModel):
    """ A 'simple' base model in the sense that its just
    a fairly standard single profiles model. Most like you'll
    inherit from this to do your own fuckery
    
    Parameters
    ----------
    name: string
        Name to use in logging
    
    planet: :obj:`Planet` or :obj:`None`
        Planet object created or None to use a default

    
    """
    def __init__(self,name,
                            planet=None,
                            star=None,
                            pressure_profile=None,
                            temperature_profile=None,
                            gas_profile=None,
                            opacities=None,
                            cia=None,
                            opacity_path=None,
                            cia_path=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,

                            ):
        super().__init__(name,opacities,cia,opacity_path,cia_path)

        self._planet = planet
        self._star=star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._gas_profile = gas_profile

        self.setup_defaults(nlayers,atm_min_pressure,atm_max_pressure)

    def setup_defaults(self,nlayers,atm_min_pressure,atm_max_pressure):
        if self._pressure_profile is None:
            from taurex.data.profiles.pressure import SimplePressureProfile
            self.info('No pressure profile defined, using simple pressure profile with')
            self.info('parameters nlayers: {}, atm_pressure_range=({},{})'.format(nlayers,atm_min_pressure,atm_max_pressure))
            self._pressure_profile = SimplePressureProfile()

 
    




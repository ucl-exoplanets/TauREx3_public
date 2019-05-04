from taurex.log import Logger
import numpy as np
import math
import pathlib
class ForwardModel(Logger):
    """A base class for producing forward models"""

    def __init__(self,name,opacities=None,
                            opacity_path=None):
        super().__init__(name)
        self.opacity_dict = {}
        self.cia_dict = {}
        

        self._opacities=opacities
        self._opacity_path = opacity_path

        self.fitting_parameters = {}

        self.contribution_list = []

    def __getitem__(self,key):
        return self.fitting_parameters[key][2]()

    def __setitem__(self,key,value):
        return self.fitting_parameters[key][3](value) 

    def add_opacity(self,opacity,molecule_filter=None):
        self.info('Reading opacity {}'.format(opacity.moleculeName))
        if opacity.moleculeName in self.opacity_dict:
            self.error('Opacity with name {} already in opactiy dictionary {}'.format(opacity.moleculeName,self.opacity_dict.keys()))
            raise Exception('Opacity for molecule {} already exists')
        
        if molecule_filter is not None:
            if opacity.moleculeName in molecule_filter:
                self.info('Loading opacity {} into model'.format(opacity.moleculeName))
                self.opacity_dict[opacity.moleculeName] = opacity       
        else:     
            self.info('Loading opacity {} into model'.format(opacity.moleculeName))
            self.opacity_dict[opacity.moleculeName] = opacity    
    def load_opacity_from_path(self,path,molecule_filter=None):
        from glob import glob
        import os
        from taurex.opacity import PickleOpacity
        glob_path = os.path.join(path,'*.pickle')

        file_list = glob(glob_path)
        self.debug('File list {}'.format(file_list))
        for files in file_list:
            splits = pathlib.Path(files).stem.split('.')
            if molecule_filter is not None:
                if not splits[0] in molecule_filter:
                    continue
            op = PickleOpacity(files)
            self.add_opacity(op,molecule_filter=molecule_filter)


    def add_contribution(self,contrib):
        from taurex.contributions import Contribution
        if not isinstance(contrib,Contribution):
            raise TypeError('Is not a a contribution type')
        else:
            if not contrib in self.contribution_list:
                self.contribution_list.append(contrib)
            else:
                raise Exception('Contribution already exists')


    def load_opacities(self,opacities=None,opacity_path=None,molecule_filter=None):
        from taurex.opacity import Opacity
        if opacities is None:
            opacities = self._opacities
        
        if opacity_path is None:
            opacity_path = self._opacity_path

        if opacities is not None:
            if isinstance(opacities,(list,)):
                self.debug('Opacity passed is list')
                for opacity in opacities:
                    self.add_opacity(opacity,molecule_filter=molecule_filter)
            elif isinstance(opacities,Opacity):
                self.add_opacity(opacities,molecule_filter=molecule_filter)
            else:
                self.error('Unknown type {} passed into opacities, should be a list, single \
                     opacity or None if reading a path'.format(type(opacities)))
                raise Exception('Unknown type passed into opacities')
        elif opacity_path is not None:

            if isinstance(opacity_path,str):
                self.load_opacity_from_path(opacity_path,molecule_filter=molecule_filter)
            elif isinstance(opacity_path,(list,)):
                for path in opacity_path:
                    self.load_opacity_from_path(path,molecule_filter=molecule_filter)
        self._opacities = None

             
    
    def build(self):
        raise NotImplementedError

    def model(self,wngrid,return_contrib=True):
        """Computes the forward model for a wngrid"""
        raise NotImplementedError


    @property
    def pressureProfile(self):
        raise NotImplementedError

    @property
    def temperatureProfile(self):
        raise NotImplementedError
    
    @property
    def altitudeProfile(self):
        raise NotImplementedError
    
    @property
    def fittingParameters(self):
        return self.fitting_parameters
    



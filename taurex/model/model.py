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
        

        self._opacities=opacities
        self._opacity_path = opacity_path

        self._cia = cia
        self._cia_path=None


        self.fitting_parameters = {}

  

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
            op = PickleOpacity(files)
            self.add_opacity(op,molecule_filter=molecule_filter)



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

    def load_cia(self,cia_xsec=None,cia_path=None):
        from taurex.cia import CIA
        if cia_xsec is None:
            cia_xsec = self._cia
        if cia_path is None:
            cia_path = self._cia_path


        if cia_xsec is not None:
            if isinstance(cia_xsec,(list,)):
                self.debug('cia passed is list')
                for xsec in cia_xsec:
                    self.add_cia(xsec)
            elif isinstance(cia_xsec,CIA):
                self.add_cia(cia_xsec)
            else:
                self.error('Unknown type {} passed into cia, should be a list, single \
                     cia or None if reading a path'.format(type(xsec)))
                raise Exception('Unknown type passed into cia')
        elif cia_path is not None:

            if isinstance(cia_path,str):
                self.load_cia_from_path(cia_path)
            elif isinstance(cia_path,(list,)):
                for path in cia_path:
                    self.load_cia_from_path(path)        
    
    def build(self):
        self.load_opacities()
        self.load_cia()


    def model(self,wngrid):
        """Computes the forward model for a wngrid"""
        raise NotImplementedError

    
    @property
    def fittingParameters(self):
        return self.fitting_parameters

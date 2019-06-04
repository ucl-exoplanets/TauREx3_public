from .singleton import Singleton
from taurex.log import Logger
import pathlib
class CIACache(Singleton):
    """Implements a lazy load of opacities"""
    def init(self):
        self.cia_dict={}
        self._cia_path = None
        self.log = Logger('CIACache')

    def set_cia_path(self,cia_path):
        self._cia_path  = cia_path



    def __getitem__(self,key):
        key = key.upper()
        if key in self.cia_dict:
            return self.cia_dict[key]
        else:
            #Try a load of the cia
            self.load_cia(pair_filter=[key])
            #If we have it after a load then good job boys
            if key in self.cia_dict:
                return self.cia_dict[key]
            else:
                #Otherwise throw an error
                self.log.error('CIA for pair %s could not be loaded',key)
                self.log.error('It could not be found in the local dictionary %s',list(self.cia_dict.keys()))
                self.log.error('Or paths %s',self._cia_path)
                self.log.error('Try loading it manually/ putting it in a path')
                raise Exception('cia could notn be loaded')

    def add_cia(self,cia,pair_filter=None):
        self.log.info('Loading cia %s into model',cia.pairName)
        if cia.pairName in self.cia_dict:
            self.log.error('cia with name %s already in opactiy dictionary %s',cia.pairName,self.cia_dict.keys())
            raise Exception('cia for molecule %s already exists')
        if pair_filter is not None:
            if cia.pairName in pair_filter:
                self.log.info('Loading cia %s into model',cia.pairName)
                self.cia_dict[cia.pairName] = cia               
        self.cia_dict[cia.pairName] = cia   
    
    def load_cia_from_path(self,path,pair_filter=None):
        from glob import glob
        from pathlib import Path
        import os
        from taurex.cia import PickleCIA

        #Find .db files
        glob_path = os.path.join(path,'*.db')

        file_list = glob(glob_path)
        self.log.debug('Glob list: %s',glob_path)
        self.log.debug('File list FOR CIA %s',file_list)
        for files in file_list:
            pairname=Path(files).stem.split('_')[0].upper()
            self.log.debug('pairname found %s',pairname)
            if pair_filter is not  None:
                if not pairname in pair_filter:
                    continue
            op = PickleCIA(files,pairname)
            self.add_cia(op)

        #Find .cia files
        glob_path = os.path.join(path,'*.cia')

        file_list = glob(glob_path)
        self.log.debug('File list %s',file_list)
        
        for files in file_list:
            from taurex.cia import HitranCIA
            pairname=Path(files).stem.split('_')[0].upper()

            if pair_filter is not  None:
                if not pairname in pair_filter:
                    continue
            op = HitranCIA(files)
            self.add_cia(op)       
            

    def load_cia(self,cia_xsec=None,cia_path=None,pair_filter=None):
        from taurex.cia import CIA
        if cia_path is None:
            cia_path = self._cia_path
        
        self.log.debug('CIA XSEC, CIA_PATH %s %s',cia_xsec,cia_path)
        if cia_xsec is not None:
            if isinstance(cia_xsec,(list,)):
                self.log.debug('cia passed is list')
                for xsec in cia_xsec:
                    self.add_cia(xsec,pair_filter=pair_filter)
            elif isinstance(cia_xsec,CIA):
                self.add_cia(cia_xsec,pair_filter=pair_filter)
            else:
                self.log.error('Unknown type %s passed into cia, should be a list, single \
                     cia or None if reading a path',type(xsec))
                raise Exception('Unknown type passed into cia')
        elif cia_path is not None:

            if isinstance(cia_path,str):
                self.load_cia_from_path(cia_path,pair_filter=pair_filter)
            elif isinstance(cia_path,(list,)):
                for path in cia_path:
                    self.load_cia_from_path(path,pair_filter=pair_filter)  
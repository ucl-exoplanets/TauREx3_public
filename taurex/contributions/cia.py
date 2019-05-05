from .contribution import Contribution
import numpy as np
import numba

@numba.jit(nopython=True,parallel=True)
def cia_numba(sigma,density,path,nlayers,ngrid,nmols,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(nlayers-layer):
        _path = path[k]
        _density = density[k+layer]
        for mol in range(nmols):
            for wn in range(ngrid):
                tau[wn] += sigma[k+layer,mol,wn]*_path*_density*_density
    return tau


class CIAContribution(Contribution):


    def __init__(self,cia=None,cia_path=None):
        super().__init__('CIA')

        self.cia_dict = {}
        self._cia = cia
        self._cia_path = cia_path

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

        #Find .db files
        glob_path = os.path.join(path,'*.db')

        file_list = glob(glob_path)
        self.debug('File list FOR CIA {}'.format(file_list))
        for files in file_list:
            pairname=Path(files).stem
            op = PickleCIA(files,pairname)
            self.add_cia(op)

        #Find .cia files
        glob_path = os.path.join(path,'*.cia')

        file_list = glob(glob_path)
        self.debug('File list {}'.format(file_list))
        for files in file_list:
            from taurex.cia import HitranCIA
            op = HitranCIA(files)
            self.add_cia(op)       
            

    def load_cia(self,cia_xsec=None,cia_path=None):
        from taurex.cia import CIA
        if cia_xsec is None:
            cia_xsec = self._cia
        if cia_path is None:
            cia_path = self._cia_path
        
        self.debug('CIA XSEC, CIA_PATH {} {}'.format(cia_xsec,cia_path))
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



    def contribute(self,model,layer,density,path_length,return_contrib):
        import numexpr as ne
        total_layers = model.pressure_profile.nLayers
        # sigma = self.sigma_cia[layer:total_layers,:]
        # combined_pt_dt = (density*density*path_length)[...,None,None]
        # contrib = ne.evaluate('sum(sigma*combined_pt_dt,axis=0)')

        # contrib = ne.evaluate('sum(contrib,axis=0)')

        self._total_contrib[layer,:]+=cia_numba(self.sigma_cia,density,path_length,self._nlayers,self._ngrid,self._total_cia,layer)

        #if return_contrib:
        #self._total_contrib[layer,:]+=contrib
        #return contrib



    def build(self,model):
        self.load_cia()
    
    def prepare(self,model,wngrid):
        total_cia = len(self.cia_dict)
        if total_cia == 0:
            return
        self.sigma_cia = np.zeros(shape=(model.pressure_profile.nLayers,total_cia,wngrid.shape[0]))

        self._total_cia = total_cia
        self._nlayers = model.pressure_profile.nLayers
        self._ngrid = wngrid.shape[0]
        self.info('Computing CIA ')
        for cia_idx,cia in enumerate(self.cia_dict.values()):
            for idx_layer,temperature in enumerate(model.temperatureProfile):
                _cia_xsec = cia.cia(temperature,wngrid)
                cia_factor = model._gas_profile.get_gas_mix_profile(cia.pairOne)*model._gas_profile.get_gas_mix_profile(cia.pairTwo)

                self.sigma_cia[idx_layer,cia_idx] = _cia_xsec*cia_factor[idx_layer]

        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))

    @property
    def totalContribution(self):
        return self._total_contrib


from taurex.log import Logger
from taurex.util import get_molecular_weight,molecule_texlabel
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.output.writeable import Writeable
import math
class GasProfile(Fittable,Logger,Writeable):
    """
    Defines gas profiles

    """

    


    def __init__(self,name,mode='linear'):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        self.active_mixratio_profile = None
        self.inactive_mixratio_profile = None

        self.active_gases = []
        self.inactive_gases = []
        self.mu_profile = None
        self.setLinearLogMode(mode)
    
    def setLinearLogMode(self,value):
        value = value.lower()
        if value in ('linear','log',):
            self._log_mode =  (value == 'log')
        else:
            raise AttributeError('Linear/Log Mode muse be either \'linear\' or \'log\'')

    @property
    def isInLogMode(self):
        return self._log_mode


    def readableValue(self,value):
        """Helper function to convert the read value to either linear or log space"""

        if self.isInLogMode:
            return math.log10(value)
        else:
            return value
    
    def writeableValue(self,value):
        """Write value back to linear space when in either log or linear mode"""
        if self.isInLogMode:
            return math.pow(10,value)
        else:
            return value

    def initialize_profile(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        self.nlayers=nlayers
        self.nlevels = nlayers+1
        self.pressure_profile = pressure_profile
        self.temperature_profile = temperature_profile
        self.altitude_profile = altitude_profile

        

        self.compute_active_gas_profile()
        self.compute_inactive_gas_profile()
        self.compute_mu_profile()
    def compute_active_gas_profile(self):
        raise NotImplementedError
    
    def compute_inactive_gas_profile(self):
        raise NotImplementedError

    def compute_mu_profile(self):
        self.mu_profile= np.zeros(shape=(self.nlayers,))
        if self.activeGasMixProfile is not None:
            for idx, gasname in enumerate(self.active_gases):
                self.mu_profile += self.activeGasMixProfile[idx,:]*get_molecular_weight(gasname)
        if self.inActiveGasMixProfile is not None:
            for idx, gasname in enumerate(self.inactive_gases):
                self.mu_profile += self.inActiveGasMixProfile[idx,:]*get_molecular_weight(gasname)
    @property
    def activeGasMixProfile(self):
        return self.active_mixratio_profile


    def get_gas_mix_profile(self,gas_name):
        if gas_name in self.activeGases:
            idx = self.active_gases.index(gas_name)
            return self.activeGasMixProfile[idx,:]
        elif gas_name in self.inActiveGases:
            idx = self.inactive_gases.index(gas_name)
            return self.inActiveGasMixProfile[idx,:]  
        else:
            raise KeyError     

    @property
    def inActiveGasMixProfile(self):
        return self.inactive_mixratio_profile

    @property
    def activeGases(self):
        return self.active_gases
    
    @property
    def inActiveGases(self):
        return self.inactive_gases

    @property
    def muProfile(self):
        return self.mu_profile

    def write(self,output):

        gas_entry = output.create_group('Gas')
        gas_entry.write_string('gas_profile_type',self.__class__.__name__)
        gas_entry.write_string_array('active_gases',self.activeGases)
        gas_entry.write_string_array('inactive_gases',self.inActiveGases)
        gas_entry.write_array('active_gas_mix_profile',self.activeGasMixProfile)
        gas_entry.write_array('inactive_gas_mix_profile',self.inActiveGasMixProfile)
        gas_entry.write_array('mu_profile',self.muProfile)
        gas_entry.write_scalar('inLogMode',self._log_mode)
        return gas_entry

class TaurexGasProfile(GasProfile):




    def __init__(self,name,active_gases,active_gas_mix_ratio,n2_mix_ratio=0,he_h2_ratio=0.17647,mode='linear'):
        super().__init__(name,mode=mode)
        if isinstance(active_gases,str):
            active_gases = [active_gases]
        
        self.active_gases = active_gases
        self.inactive_gases = ['H2', 'HE', 'N2']
        self.debug('Active Gases {}'.format(self.active_gases))
        total_size_gases = len(self.active_gases)
        self.debug('Active gas length: {}'.format(total_size_gases))
        self.active_gas_mix_ratio = [0.0]*total_size_gases
        self.debug('Active gas mix: {}'.format(self.active_gas_mix_ratio))

        min_size = min(total_size_gases,len(active_gas_mix_ratio))

        self.active_gas_mix_ratio[:min_size] = active_gas_mix_ratio[:min_size]
        
        self.debug('Active gases {}'.format(self.active_gases))
        self.debug('Active gas mix ratio: {}'.format(self.active_gas_mix_ratio))
        self._n2_mix_ratio = n2_mix_ratio
        self._he_h2_mix_ratio = he_h2_ratio
    

    def compute_active_gas_profile(self):
        self.active_mixratio_profile=np.zeros((len(self.active_gases), self.nlayers))
        for idx,ratio in enumerate(self.active_gas_mix_ratio):
            self.active_mixratio_profile[idx, :] = ratio
    

    @fitparam(param_name='N2',param_latex=molecule_texlabel('N2'),default_fit=False,default_bounds=[1e-12,1.0])
    def N2MixRatio(self):
        return self.readableValue(self._n2_mix_ratio)
    
    @N2MixRatio.setter
    def N2MixRatio(self,value):
        self._n2_mix_ratio = self.writeableValue(value)

    @fitparam(param_name='H2_He',param_latex=molecule_texlabel('H$_2$/He'),default_fit=False,default_bounds=[1e-12,1.0])
    def H2HeMixRatio(self):
        return self.readableValue(self._he_h2_mix_ratio)
    
    @H2HeMixRatio.setter
    def H2HeMixRatio(self,value):
        self._he_h2_mix_ratio = self.writeableValue(value)


    def compute_inactive_gas_profile(self):

        self.inactive_mixratio_profile = np.zeros((len(self.inactive_gases), self.nlayers))
        self.inactive_mixratio_profile[2, :] = self._n2_mix_ratio
        # first get the sum of the mixing ratio of all active gases
        if len(self.active_gases) > 1:
            active_mixratio_sum = np.sum(self.active_mixratio_profile, axis = 0)
        else:
            active_mixratio_sum = np.copy(self.active_mixratio_profile)
        
        active_mixratio_sum += self.inactive_mixratio_profile[2, :]
        
        mixratio_remainder = 1. - active_mixratio_sum
        self.inactive_mixratio_profile[0, :] = mixratio_remainder/(1. + self._he_h2_mix_ratio) # H2
        self.inactive_mixratio_profile[1, :] =  self._he_h2_mix_ratio * self.inactive_mixratio_profile[0, :] 


    def add_active_gas_param(self,idx):
        mol_name = self.active_gases[idx]
        param_name = mol_name
        param_tex = molecule_texlabel(mol_name)
        if self.isInLogMode:
            param_name = 'log_{}'.format(mol_name)
            param_tex = 'log({})'.format(molecule_texlabel(mol_name))
        
        def read_mol(self,idx=idx):
            return self.readableValue(self.active_gas_mix_ratio[idx])
        def write_mol(self,value,idx=idx):
            self.active_gas_mix_ratio[idx] = self.writeableValue(value)

        fget = read_mol
        fset = write_mol
        
        bounds = [1.0e-12, 0.1]
        if self.isInLogMode:
            bounds=[-12,-1]
        
        default_fit = False
        self.add_fittable_param(param_name,param_tex,fget,fset,default_fit,bounds)  

    def write(self,output):

        gas_entry = super().write(output)
        gas_entry.write_array('active_gas_mix_ratios',self.active_gas_mix_ratio)
        gas_entry.write_scalar('n2_mix_ratio',self._n2_mix_ratio)
        gas_entry.write_scalar('he_h2_ratio',self._he_h2_mix_ratio)

        return gas_entry
class ComplexGasProfile(TaurexGasProfile):

    def __init__(self,
                name,
                active_gases,
                active_gas_mix_ratio,
                active_complex_gases,
                active_gases_mixratios_surface,
                active_gases_mixratios_top,
                n2_mix_ratio=0,he_h2_ratio=0.17647,mode='linear'):
        super().__init__(name,active_gases,active_gas_mix_ratio,n2_mix_ratio,he_h2_ratio,mode)
        self.active_complex_gases = active_complex_gases
        self.active_gases_mixratios_surface = active_gases_mixratios_surface
        self.active_gases_mixratios_top = active_gases_mixratios_top

        self.add_noncomplex_params()
        self.add_surface_param()
        self.add_top_param()

    def add_surface_param(self):
        for idx,mol_name in enumerate(self.active_complex_gases):
            mol_tex = molecule_texlabel(mol_name)

            param_surface = 'S {}'.format(mol_name)
            param_surf_tex = '{}'.format(mol_tex)

            if self.isInLogMode:
                param_surface = 'S_log_{}'.format(mol_name)
                param_surf_tex = 'S_log({})'.format(mol_tex)

            def read_surf(self,idx=idx):
                return self.readableValue(self.active_gases_mixratios_surface[idx])
            def write_surf(self,value,idx=idx):
                self.active_gases_mixratios_surface[idx] = self.writeableValue(value)

            fget_surf = read_surf
            fset_surf = write_surf

            bounds = [1.0e-12, 0.1]
            if self.isInLogMode:
                bounds=[-12,-1]

            default_fit = False
            self.add_fittable_param(param_surface,param_surf_tex ,fget_surf,fset_surf,default_fit,bounds)   

    def add_top_param(self):
        for idx,mol_name in enumerate(self.active_complex_gases): 
            mol_tex = molecule_texlabel(mol_name)

            param_top = 'T {}'.format(mol_name)
            param_top_tex = '{}'.format(mol_tex)

            if self.isInLogMode:
                param_top = 'T_log_{}'.format(mol_name)
                param_top_tex = 'T_log({})'.format(mol_tex)

            def read_top(self,idx=idx):
                return self.readableValue(self.active_gases_mixratios_top[idx])
            def write_top(self,value,idx=idx):
                self.active_gases_mixratios_top[idx] = self.writeableValue(value)

            fget_top = read_top
            fset_top = write_top

            bounds = [1.0e-12, 0.1]
            if self.isInLogMode:
                bounds=[-12,-1]

            default_fit = False
            self.add_fittable_param(param_top,param_top_tex ,fget_top,fset_top,default_fit,bounds)      

    def add_noncomplex_params(self):

        for idx,gas in enumerate(self.active_gases):
            if not gas in self.active_complex_gases:
                self.add_active_gas_param(idx) 


    def compute_active_gas_profile(self):
        super().compute_active_gas_profile()
        self.compute_complex_gas_profile()

    
    def compute_complex_gas_profile(self):
        """Overload to compute complex gas profiles"""
        raise NotImplementedError

    
    def complex_gas_iter(self):
        """Helper function to get indices for the complex gases"""
        for j,complex_gas in enumerate(self.active_complex_gases):
            for i,active_gas in enumerate(self.active_gases):
                if complex_gas==active_gas:
                    yield i,j

    def write(self,output):

        gas_entry = super().write(output)

        gas_entry.write_string_array('active_complex_gases',self.active_complex_gases)
        gas_entry.write_array('active_gases_mixratios_surface',self.active_gases_mixratios_surface)
        gas_entry.write_array('active_gases_mixratios_top',self.active_gases_mixratios_top)

        return gas_entry
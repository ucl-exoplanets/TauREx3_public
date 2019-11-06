 
from .contribution import Contribution, contribute_tau
import numpy as np
from taurex.data.fittable import fitparam



class LeeMieContribution(Contribution):


    def __init__(self, mie_radius=0.01, mie_Q=40,
                 mie_mix_ratio=1e-10, mie_bottom_pressure=-1,
                 mie_top_pressure=-1):
        super().__init__('Mie')

        self._mie_radius = mie_radius
        self._mie_q = mie_Q
        self._mie_mix = mie_mix_ratio
        self._mie_bottom_pressure = mie_bottom_pressure
        self._mie_top_pressure = mie_top_pressure

    @fitparam(param_name='mie_radius', param_latex='$R_\mathrm{mie}$',
              default_fit=False, default_bounds=[0.01, 0.5])
    def mieRadius(self):
        return self._mie_radius

    @mieRadius.setter
    def mieRadius(self, value):
        self._mie_radius = value
    
    @fitparam(param_name='mie_q', param_latex='$Q_\mathrm{ext}$',
              default_fit=False, default_bounds=[-10, 1])
    def mieQ(self):
        return self._mie_q
    
    @mieQ.setter
    def mieQ(self, value):
        self._mie_q = value  

    @fitparam(param_name='mie_topP',param_latex='$P^{mie}_\mathrm{top}$',default_mode='log',default_fit=False,default_bounds=[-1,1])
    def mieTopPressure(self):
        return self._mie_top_pressure
    
    @mieTopPressure.setter
    def mieTopPressure(self, value):
        self._mie_top_pressure = value

    @fitparam(param_name='mie_bottomP',param_latex='$P^{mie}_\mathrm{bottom}$',default_mode='log',default_fit=False,default_bounds=[-1,1])
    def mieBottomPressure(self):
        return self._mie_bottom_pressure
    
    @mieBottomPressure.setter
    def mieBottomPressure(self, value):
        self._mie_bottom_pressure = value

    @fitparam(param_name='mie_mix_ratio',param_latex='$\chi_\mathrm{mie}$',default_mode='log',default_fit=False,default_bounds=[-1,1])
    def mieMixing(self):
        return self._mie_mix
    
    @mieMixing.setter
    def mieMixing(self, value):
        self._mie_mix = value



    def build(self, model):
        pass
    

    def finalize(self, model):
        raise NotImplementedError

    def prepare_each(self, model, wngrid):
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_profile = model.pressureProfile

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:
            bottom_pressure = pressure_profile[0]

        top_pressure = self.mieTopPressure
        if top_pressure < 0:
            top_pressure = pressure_profile[-1]       

        wltmp = 10000/wngrid

        a = self.mieRadius

        x = 2.0 * np.pi * a / wltmp
        self.debug('wngrid %s', wngrid)
        self.debug('x %s', x)
        Qext = 5.0 / (self.mieQ * x**(-4.0) + x**(0.2))

        sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))

        am=a*1e-6

        sigma_mie = Qext * np.pi * (am**2.0)

        self.debug('Qext %s', Qext)
        self.debug('radius um %s', a)
        self.debug('sigma %s', sigma_mie)

        self.debug('bottome_pressure %s',bottom_pressure)
        self.debug('top_pressure %s',top_pressure)

        
        cloud_filter = (pressure_profile <= bottom_pressure) & (pressure_profile >= top_pressure)

        sigma_xsec[cloud_filter, ...] = sigma_mie 

        self.sigma_xsec = sigma_xsec * self.mieMixing

        self.debug('final xsec %s', self.sigma_xsec[:, :])
        self.debug('final xsec %s', self.sigma_xsec.max())
        
        #self._total_contrib[...]=0.0
        yield 'Lee', sigma_xsec


    def write(self,output):
        contrib = super().write(output)
        contrib.write_scalar('mie_radius', self._mie_radius)
        contrib.write_scalar('mie_Q', self._mie_q)
        contrib.write_scalar('mie_mix_ratio', self._mie_mix)
        contrib.write_scalar('mie_bottom_pressure', self._mie_bottom_pressure)
        contrib.write_scalar('mie_top_pressure', self._mie_top_pressure)
        return contrib
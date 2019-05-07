import sys
import os
import logging
logging.basicConfig(level=logging.ERROR)
import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/ahmed/Documents/repos/TauREx3/')
from taurex.model.transmission import TransmissionModel
from taurex.data.profiles.gas import ConstantGasProfile
from taurex.data.profiles.temperature import Isothermal
from taurex.contributions import *
import numpy as np
import time
from taurex.cache import OpacityCache,CIACache
OpacityCache().set_opacity_path('/Users/ahmed/Documents/taurex_files/xsec/TauRex_sampled_xsecs_R10000_0.3-15')
CIACache().set_cia_path('/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/cia/hitran/')


gas = ConstantGasProfile(active_gases=['H2O'],
                        active_gas_mix_ratio=[1e-4],n2_mix_ratio=1e-12,mode='log')

temp = Isothermal(iso_temp=1550)

tm = TransmissionModel(nlayers=30,gas_profile= gas,temperature_profile=temp
                ,atm_min_pressure=1e-5,atm_max_pressure=1e6)
tm.add_contribution(CIAContribution(cia_pairs=['H2-He','H2-H2']))
tm.add_contribution(RayleighContribution())
tm.build()

from taurex.optimizer.multinest import MultiNestOptimizer

from taurex.optimizer.nestle import NestleOptimizer
from taurex.data.spectrum.observed import ObservedSpectrum

opt = MultiNestOptimizer('/Users/ahmed/Documents/taurex_files/multinest',model=tm)
#opt = NestleOptimizer(model=tm)
obs = ObservedSpectrum('/Users/ahmed/Documents/taurex_files/taurex_cobweb/tests/test_0_transmission/SPECTRUM_fit.dat')



print(tm.fittingParameters.keys())

opt.set_observed(obs)

opt.set_wavenumber_grid(obs.wavenumberGrid)

opt.compile_params()
opt.enable_fit('T')
opt.set_boundary('T',[1300.0, 1800.0])
opt.enable_fit('planet_radius')
opt.enable_fit('N2')
opt.enable_fit('H2_He')
opt.enable_fit('log_H2O')


opt.set_boundary('log_H2O',[-12.0, 12.0])
opt.set_boundary('N2',[-12.0, 0.0])
opt.set_boundary('H2_He',[-12.0, 0.0])

opt.compile_params()

print(opt.fit_names)
print(opt.fit_values)
print(opt.fit_boundaries)
#quit()


start = time.time()
opt.fit()

end = time.time()-start

print('Fitting took {} seconds '.format(end))

#opt.fit()
#[('planet_radius', 0.9977460143147965), ('T', 1773.753290656476), ('N2', -2.9852926824091544), ('H2_He', 11.611379020348924), ('log_H2O', -3.6689485064871157)]
#[('planet_radius', 0.9954708431819737), ('T', 1680.2792218153766), ('N2', -1.2510885714329305), ('H2_He', -7.606159994370432), ('log_H2O', -3.327989739664458)]
print(list(zip(opt.fit_names,opt.fit_values)))


obs_bins = obs.wavenumberGrid
xsec_wnbins = OpacityCache()['H2O'].wavenumberGrid
absptn,tau,contrib = tm.model(xsec_wnbins)

bin_means = (np.histogram(xsec_wnbins,obs_bins, weights=absptn)[0] /
             np.histogram(xsec_wnbins,obs_bins)[0])

wlgrid = np.log10(obs.wavelengthGrid)

plt.plot(wlgrid,obs.spectrum)
plt.plot(wlgrid[:-1],bin_means)

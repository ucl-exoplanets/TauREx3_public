import sys
import os
import logging
logging.basicConfig(level=logging.ERROR)
import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/ahmed/Documents/repos/TauREx3/')
from taurex.model import TransmissionModel
from taurex.model.lightcurve import LightCurveModel
from taurex.data.profiles.chemistry import TaurexChemistry,ConstantGas
from taurex.data.profiles.temperature import Isothermal
from taurex.data.stellar import BlackbodyStar
from taurex.data.spectrum.lightcurve import ObservedLightCurve
from taurex.data.planet import Planet
from taurex.contributions import *
import time
import numpy as np

from taurex.cache import OpacityCache,CIACache
OpacityCache().set_opacity_path('/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/xsec/TauRex_sampled_xsecs_R10000_0.3-15')
CIACache().set_cia_path('/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/cia/hitran/')



star = BlackbodyStar(radius=1.155,temperature=6092.0)
temp_planet = Isothermal(iso_temp=1500.0)
chemistry = TaurexChemistry(he_h2_ratio=0.17647)
chemistry.addGas(ConstantGas('H2O',mix_ratio=1e-6))
planet = Planet()


tm = TransmissionModel(planet=planet,star=star,temperature_profile=temp_planet,chemistry=chemistry,nlayers=100)
tm.add_contribution(AbsorptionContribution())
#tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
tm.add_contribution(RayleighContribution())
tm.add_contribution(SimpleCloudsContribution(clouds_pressure=1e3))


lc = LightCurveModel(tm,'/Users/ahmed/Downloads/lc_input_correct_wfc3.pickle',False,True,False,True,True)
lc.build()

from taurex.optimizer.multinest import MultiNestOptimizer

#from taurex.optimizer.nestle import NestleOptimizer
from taurex.data.spectrum.observed import ObservedSpectrum

opt = MultiNestOptimizer(model=lc
                ,num_live_points=1000,evidence_tolerance=.001,search_multi_modes=False,
                multi_nest_path='/Users/ahmed/Documents/repos/TauREx3/examples/multinest')
#opt = NestleOptimizer(model=tm)
obs = ObservedLightCurve('/Users/ahmed/Downloads/lc_input_correct_wfc3.pickle')


opt.set_observed(obs)

opt.set_wavenumber_grid(obs.wavenumberGrid)

opt.compile_params()
opt.enable_fit('T')
opt.enable_fit('planet_radius')
opt.disable_fit('N2')
opt.disable_fit('H2_He')
opt.enable_fit('H2O')
opt.enable_fit('clouds_pressure')
opt.set_boundary('T',[1000.0, 2000.0])
opt.set_boundary('H2O',[1.0e-12, 1.0e-1])
opt.set_boundary('planet_radius',[ 0.5, 2.0])


opt.compile_params()



print(opt.fit_names)
print(opt.fit_values)
print(opt.fit_boundaries)
#quit()

start = time.time()
opt.fit()

end = time.time()-start

print('Fitting took {} seconds '.format(end))

# #opt.fit()
# #[('planet_radius', 0.9977460143147965), ('T', 1773.753290656476), ('N2', -2.9852926824091544), ('H2_He', 11.611379020348924), ('log_H2O', -3.6689485064871157)]
# #[('planet_radius', 0.9954708431819737), ('T', 1680.2792218153766), ('N2', -1.2510885714329305), ('H2_He', -7.606159994370432), ('log_H2O', -3.327989739664458)]
print(list(zip(opt.fit_names,opt.fit_values)))


# quit()
# obs_bins = obs.wavenumberGrid
# xsec_wnbins = OpacityCache()['H2O'].wavenumberGrid
# absptn,tau,contrib = tm.model(xsec_wnbins)

# bin_means = (np.histogram(xsec_wnbins,obs_bins, weights=absptn)[0] /
#              np.histogram(xsec_wnbins,obs_bins)[0])

# wlgrid = np.log10(obs.wavelengthGrid)

# plt.plot(wlgrid,obs.spectrum)
# plt.plot(wlgrid[:-1],bin_means)

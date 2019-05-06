
import numpy as np
import sys,os
sys.path.insert(0,'../')
os.environ['NUMBAPRO_NVVM'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\nvvm\\bin\\nvvm64_32_0.dll"
os.environ['NUMBAPRO_LIBDEVICE'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\nvvm\\libdevice"

#print(sys.path)
from taurex.data.profiles.gas import ConstantGasProfile
from taurex.data.profiles.pressure import SimplePressureProfile
from taurex.data.profiles.temperature import Guillot2010,Isothermal
from taurex.data.planet import Earth,Planet

from taurex.cache import OpacityCache

import time
import matplotlib
from taurex.data.stellar import Star
from taurex.constants import RSOL,MJUP,RJUP
import matplotlib.pyplot as plt
from taurex.model import TransmissionModel
#from taurex.contributions.cuda.absorption import GPUAbsorptionContribution
#from taurex.contributions.cuda.cia import GPUCIAContribution
#from taurex.contributions.cuda.rayleigh import GPURayleighContribution
from taurex.contributions import CIAContribution,RayleighContribution,AbsorptionContribution
import logging
import numexpr as ne


logging.basicConfig(level=logging.INFO)

OpacityCache().set_opacity_path('/Users/ahmed/Documents/taurex_files/xsec/TauRex_sampled_xsecs_R10000_0.3-15')

absc = AbsorptionContribution()

tm = TransmissionModel(gas_profile=ConstantGasProfile(active_gases=['H2O'],active_gas_mix_ratio=[1e-6]),
                       planet=Planet(),
                       star=Star(),
                       temperature_profile=Isothermal(),nlayers=30,
                        abs_contrib=absc)

tm.add_contribution(CIAContribution(cia_path='/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/cia/hitran/'))
tm.add_contribution(RayleighContribution())

tm.build()

wngrid = OpacityCache()['H2O'].wavenumberGrid

absorption,tau,contributions = tm.model(wngrid,return_contrib=True)

start = time.time()
for x in range(10):
    tm.model(wngrid,return_contrib=True)

end = time.time()

print('Total time taken for 10 iterations {} s time per iteration {}'.format(end-start,(end-start)/10))

wlgrid = np.log10(10000/wngrid)

fig = plt.figure()

for name,value in contributions:
    plt.plot(wlgrid,value,label=name)



plt.plot(wlgrid,absorption,label='total')
plt.legend()
plt.show()













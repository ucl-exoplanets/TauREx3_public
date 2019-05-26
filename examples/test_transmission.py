
import numpy as np
import sys,os
sys.path.insert(0,'../')
os.environ['NUMBAPRO_NVVM'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\nvvm\\bin\\nvvm64_32_0.dll"
os.environ['NUMBAPRO_LIBDEVICE'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\nvvm\\libdevice"

#print(sys.path)
from taurex.data.profiles.pressure import SimplePressureProfile
from taurex.data.profiles.temperature import Guillot2010,Isothermal
from taurex.data.planet import Earth,Planet

from taurex.cache import OpacityCache,CIACache

import time
import matplotlib
from taurex.data.stellar import BlackbodyStar
from taurex.constants import RSOL,MJUP,RJUP
import matplotlib.pyplot as plt
from taurex.model import EmissionModel,TransmissionModel

from taurex.data.profiles.chemistry import TaurexChemistry,ConstantGas

from taurex.contributions import CIAContribution,RayleighContribution,AbsorptionContribution
import logging
import numexpr as ne


logging.basicConfig(level=logging.INFO)

OpacityCache().set_opacity_path('/Users/ahmed/Documents/taurex_files/xsec/TauRex_sampled_xsecs_R10000_0.3-15')
CIACache().set_cia_path('/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/cia/hitran/')
absc = AbsorptionContribution()



tm = TransmissionModel(
                       planet=Planet(),
                       star=BlackbodyStar(temperature=5800),
                       temperature_profile=Isothermal(),nlayers=30)
tm.add_contribution(AbsorptionContribution())
#tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
#tm.add_contribution(RayleighContribution())

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













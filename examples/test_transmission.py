
import numpy as np
import sys
sys.path.insert(0,'../')
#print(sys.path)
from taurex.data.profiles.gas import ConstantGasProfile
from taurex.data.profiles.pressure import SimplePressureProfile
from taurex.data.profiles.temperature import Guillot2010,Isothermal
from taurex.data.planet import Earth,Planet

import time

from taurex.data.stellar import Star
from taurex.constants import RSOL,MJUP,RJUP
import matplotlib.pyplot as plt
from taurex.model import TransmissionModel
from taurex.contributions import CIAContribution,RayleighContribution
import logging
import numexpr as ne


logging.basicConfig(level=logging.INFO)

tm = TransmissionModel(gas_profile=ConstantGasProfile(active_gases=['H2O','CH4'],active_gas_mix_ratio=[1e-6,1e-4]),
                       opacity_path='C:/Users/Bahamut/Documents/TaurexFiles/Input/xsec/TauRex_sampled_xsecs_R10000_0.3-15',
                       planet=Planet(),
                       star=Star(),
                       temperature_profile=Isothermal())

tm.add_contribution(CIAContribution(cia_path='C:/Users/Bahamut/Documents/TaurexFiles/Input/cia/hitran/'))
tm.add_contribution(RayleighContribution())

tm.build()

wngrid = np.linspace(100,30000,30000)


absorption,tau,contributions = tm.model(wngrid)

start = time.time()
for x in range(10):
    tm.model(wngrid)

end = time.time()

print('Total time taken for 10 iterations {} s time per iteration {}'.format(end-start,(end-start)/10))

wlgrid = np.log10(10000/wngrid)

fig = plt.figure()

for name,value in contributions:
    plt.plot(wlgrid,value,label=name)

plt.plot(wlgrid,absorption,label='total')
plt.legend()
plt.show()













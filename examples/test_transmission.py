import taurex
import numpy as np
from taurex.data.profiles.gas import ConstantGasProfile
from taurex.data.profiles.pressure import SimplePressureProfile
from taurex.data.profiles.temperature import Guillot2010,Isothermal
from taurex.data.planet import Earth,Planet

from taurex.data.stellar import Star
from taurex.constants import RSOL,MJUP,RJUP
import matplotlib.pyplot as plt
from taurex.model import TransmissionModel
from taurex.contributions import CIAContribution,RayleighContribution
import logging
import numexpr as ne

ne.set_num_threads(4)

logging.basicConfig(level=logging.INFO)

tm = TransmissionModel(gas_profile=ConstantGasProfile(n2_mix_ratio=0.8,active_gases=['CO'],active_gas_mix_ratio=[0.2]),
                       opacity_path='/Users/ahmed/Documents/taurex_files/xsec/TauRex_sampled_xsecs_R10000_0.3-15/',
                       planet=Planet(radius=0.1*RJUP,mass=0.01*MJUP),
                       star=Star(radius=0.35*RSOL),
                       temperature_profile=Isothermal(iso_temp=300.0))

tm.add_contribution(CIAContribution(cia_path='/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/cia/hitran/'))
tm.add_contribution(RayleighContribution())

tm.build()

wngrid = np.linspace(100,30000,30000)

absorption,tau,contributions = tm.model(wngrid)

wlgrid = np.log10(10000/wngrid)

fig = plt.figure()

for name,value in contributions:
    plt.plot(wlgrid,value,label=name)

plt.plot(wlgrid,absorption,label='total')
plt.legend()
plt.show()













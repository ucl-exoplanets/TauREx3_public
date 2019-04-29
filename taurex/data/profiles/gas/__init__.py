from .gasprofile import TaurexGasProfile,GasProfile,ComplexGasProfile
from .constantprofile import ConstantGasProfile
from .twopointgasprofile import TwoPointGasProfile
try:
    from .acegasprofile import ACEGasProfile
except:
    print('ACE not detected')
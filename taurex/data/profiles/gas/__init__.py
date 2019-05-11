from .gasprofile import TaurexGasProfile,GasProfile,ComplexGasProfile
from .constantprofile import ConstantGasProfile
from .twopointgasprofile import TwoPointGasProfile
from .twolayergasprofile import TwoLayerGasProfile
try:
    from .acegasprofile import ACEGasProfile
except:
    print('ACE not detected')
from taurex.data.spectrum.spectrum import BaseSpectrum
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.data.spectrum.array import ArraySpectrum
from taurex.data.spectrum.iraclis import IraclisSpectrum
try:
    from taurex.data.spectrum.lightcurve import ObservedLightCurve
except ImportError:
    print('pylightcurve not install. Ignoring')
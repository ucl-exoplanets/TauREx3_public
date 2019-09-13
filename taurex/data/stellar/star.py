from taurex.log import Logger
from taurex.constants import G,RJUP,MJUP,RSOL
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.util.emission import black_body
from taurex.output.writeable import Writeable
class BlackbodyStar(Fittable,Logger,Writeable):
    """
    A base class that holds information on the star in the model.
    Its implementation is a star that has a blackbody spectrum.


    Parameters
    -----------
    temperature : float
        Blackbody temperature in Kelvin
    
    radius : float
        Stellar radius in terms of Solar radius


    """
    

    def __init__(self,temperature=5000,radius=1.0, distance = 1,
                  magnitudeK = 10.0,mass = 1.0,
                      ):
        Logger.__init__(self,'Star')
        Fittable.__init__(self)
        self._temperature = temperature
        self._radius = radius*RSOL
        self._mass = mass
        self.sed = None
        self.distance = distance
        self.magnitudeK = magnitudeK
    @property
    def radius(self):
        """
        Radius in metres

        Returns
        -------
        R : float

        """
        return self._radius
    
    @property
    def temperature(self):
        """
        Blackbody temperature in Kelvin

        Returns
        -------
        T : float

        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value



    @property
    def mass(self):
        return self._mass

    def initialize(self,wngrid):
        """
        Initializes the blackbody spectrum on the given wavenumber grid

        Parameters
        ----------
        wngrid : :obj:`array`
            Wavenumber grid cm-1 to compute black body spectrum
        
        """
        self.sed = black_body(wngrid,self.temperature)
    

    @property
    def spectralEmissionDensity(self):
        """
        Spectral emmision density

        Returns
        -------
        sed : :obj:`array`
        """
        return self.sed


    def write(self,output):
        star = output.create_group('Star')
        star.write_string('star_type',self.__class__.__name__)
        star.write_scalar('temperature',self.temperature)
        star.write_scalar('radius',self._radius)
        star.write_scalar('radius_RSOL',self.radius/RSOL)
        star.write_array('SED',self.spectralEmissionDensity)
        return star


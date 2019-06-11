from .singleton import Singleton
from taurex.log import Logger
import pathlib
class OpacityCache(Singleton):
    """
    Implements a lazy load of opacities. A singleton that
    loads and caches xsections as they are needed. Calling

    >>> opt = OpacityCache()
    >>> opt2 = OpacityCache()

    Reveals that:

    >>> opt == opt2
    True
    
    Importantly this class will automatically search directories for cross-sections
    set using the :func:`set_opacity_path` method:

    >>> opt.set_opacity_path('path/to/crossections')

    Currently only :obj:`.pickle` files are supported.

    To get the cross-section object for a particular molecule use the square bracket operator:

    >>> opt['H2O']
    <taurex.opacity.pickleopacity.PickleOpacity at 0x107a60be0>

    This returns a :class:`~taurex.opacity.pickleopacity.PickleOpacity` object for you to compute H2O cross sections from.
    When called for the first time, a directory search is performed and, if found, the appropriate cross-section is loaded. Subsequent calls will immediately
    return the already loaded object:

    >>> h2o_a = opt['H2O']
    >>> h2o_b = opt['H2O']
    >>> h2o_a == h2o_b
    True

    """
    def init(self):
        self.opacity_dict={}
        self._opacity_path = None
        self.log = Logger('OpacityCache')
        self._default_interpolation = 'exp'

    def set_opacity_path(self,opacity_path):
        """
        Set the path that will be searched for cross-sections. Cross-section in this path
        must be *.pickle* files and must have names of the form:

        - ``Molecule Name``.Whatever.pickle

        For H2O:

            - ``H2O.R1000.pickle``
            - ``H2O.pickle``
            - ``H2O.R1000.xxx420summergirllovesgreenday420xxx.pickle``
        
        Are all valid

        """
        

        self._opacity_path  = opacity_path
    
    def set_interpolation(self,interpolation_mode):
        """
        Sets the interpolation mode for all currently loaded (and future loaded) cross-sections

        Can either be ``linear`` for linear interpolation of both temeprature and pressure:

        >>> OpacityCache().set_interpolation('linear')

        or ``exp`` for natural exponential interpolation of temperature
        and linear for pressure

        >>> OpacityCache().set_interpolation('exp')
        

        Parameters
        ----------
        interpolation_mode: str
            Either ``linear`` for bilinear interpolation or ``exp`` for exp-linear interpolation
        


        """

        self._default_interpolation = interpolation_mode
        for values in self.opacity_dict.values():
            values.set_interpolation_mode(self._default_interpolation)

    def __getitem__(self,key):
        key = key.upper()
        if key in self.opacity_dict:
            return self.opacity_dict[key]
        else:
            #Try a load of the opacity
            self.load_opacity(molecule_filter=[key])
            #If we have it after a load then good job boys
            if key in self.opacity_dict:
                return self.opacity_dict[key]
            else:
                #Otherwise throw an error
                self.log.error('Opacity for molecule %s could not be loaded',key)
                self.log.error('It could not be found in the local dictionary %s',list(self.opacity_dict.keys()))
                self.log.error('Or paths %s',self._opacity_path)
                self.log.error('Try loading it manually/ putting it in a path')
                raise Exception('Opacity could notn be loaded')



    def add_opacity(self,opacity,molecule_filter=None):
        self.log.info('Reading opacity %s',opacity.moleculeName)
        if opacity.moleculeName in self.opacity_dict:
            self.log.warning('Opacity with name %s already in opactiy dictionary %s',opacity.moleculeName,self.opacity_dict.keys())
            return
        if molecule_filter is not None:
            if opacity.moleculeName in molecule_filter:
                self.log.info('Loading opacity %s into model',opacity.moleculeName)
                self.opacity_dict[opacity.moleculeName] = opacity       
        else:     
            self.log.info('Loading opacity %s into model',opacity.moleculeName)
            self.opacity_dict[opacity.moleculeName] = opacity    
    def load_opacity_from_path(self,path,molecule_filter=None):
        from glob import glob
        import os
        from taurex.opacity import PickleOpacity
        glob_path = os.path.join(path,'*.pickle')

        file_list = glob(glob_path)
        self.log.debug('File list %s',file_list)
        for files in file_list:
            splits = pathlib.Path(files).stem.split('.')
            if molecule_filter is not None:
                if not splits[0] in molecule_filter:
                    continue
            op = PickleOpacity(files,interpolation_mode=self._default_interpolation)
            op._molecule_name = splits[0]
            self.add_opacity(op,molecule_filter=molecule_filter)

    def load_opacity(self,opacities=None,opacity_path=None,molecule_filter=None):
        """
        """
        from taurex.opacity import Opacity
        
        if opacity_path is None:
            opacity_path = self._opacity_path

        if opacities is not None:
            if isinstance(opacities,(list,)):
                self.log.debug('Opacity passed is list')
                for opacity in opacities:
                    self.add_opacity(opacity,molecule_filter=molecule_filter)
            elif isinstance(opacities,Opacity):
                self.add_opacity(opacities,molecule_filter=molecule_filter)
            else:
                self.log.error('Unknown type %s passed into opacities, should be a list, single \
                     opacity or None if reading a path',type(opacities))
                raise Exception('Unknown type passed into opacities')
        elif opacity_path is not None:

            if isinstance(opacity_path,str):
                self.load_opacity_from_path(opacity_path,molecule_filter=molecule_filter)
            elif isinstance(opacity_path,(list,)):
                for path in opacity_path:
                    self.load_opacity_from_path(path,molecule_filter=molecule_filter)
    
    def clear_cache(self):
        """
        Clears all currently loaded cross-sections
        """
        self.opacity_dict = {}
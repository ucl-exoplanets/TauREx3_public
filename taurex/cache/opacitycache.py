"""
Contains caching class for Molecular cross section files
"""

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

    Multiple paths can be set as well

    >>> opt.set_opacity_path(['/path/to/crosssections','/another/path/to/crosssections'])

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
    
    Lastly if you've got a hot new opacity format that doesn't suck balls you can try out
    this shit by manually adding it into the cache:
    
    >>> new_h2o = MyNewOpacityFormat()
    >>> opt.add_opacity(new_h2o)
    >>> opt['H2O]
    <MyNewOpacityFormat at 0x107a60be0>

    Now Taurex will use it instead in all calculations!

    """
    def init(self):
        self.opacity_dict={}
        self._opacity_path = None
        self.log = Logger('OpacityCache')
        self._default_interpolation = 'exp'

    def set_opacity_path(self,opacity_path):
        """
        Set the path(s) that will be searched for cross-sections. Cross-section in this path
        must be *.pickle* files and must have names of the form:

        - ``Molecule Name``.Whatever.pickle

        For H2O:

            - ``H2O.R1000.pickle``
            - ``H2O.pickle``
            - ``H2O.R1000.xxx420summergirllovesgreenday420xxx.pickle``
        
        Are all valid

        Parameters
        ----------

        opacity_path : str or :obj:`list` of str, optional
            search path(s) to look for molecular opacities



        """
        
        import os
        if not os.path.isdir(opacity_path):
            self.log.error('PATH: %s does not exist!!!',opacity_path)
            raise NotADirectoryError
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
        """
        For a molecule return the relevant :class:`~taurex.opacity.opacity.Opacity` object.


        Parameter
        ---------
        key : str
            molecule name

        Returns
        -------
        :class:`~taurex.opacity.pickleopacity.PickleOpacity`
            Cross-section object desired
        
        Raise
        -----
        Exception
            If molecule could not be loaded/found

        """
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
        """

        Adds a :class:`~taurex.opacity.opacity.Opacity` object to the cache to then be
        used by Taurex 3

        Parameters
        ----------
        opacity : :class:`~taurex.opacity.opacity.Opacity`
            Opacity object to add to the cache
        
        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity object will only be included
            if its molecule is in the list. Mostly used by the 
            :func:`__getitem__` for filtering

        """
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


    def find_list_of_molecules(self):
        from glob import glob
        import os
        from taurex.opacity import PickleOpacity
        if self._opacity_path is None:
            return []
        glob_path = os.path.join(self._opacity_path,'*.pickle')

        file_list = glob(glob_path)
        self.log.debug('File list %s',file_list)
        molecules = []
        for files in file_list:
            splits = pathlib.Path(files).stem.split('.')       
            molecules.append(splits[0].upper())
        return molecules
    def load_opacity_from_path(self,path,molecule_filter=None):
        """
        Searches path for molecular cross-section files, creates and loads them into the cache
        ``.pickle`` will be loaded as :class:`~taurex.opacity.pickleopacity.PickleOpacity`
        

        Parameters
        ----------
        path : str
            Path to search for molecular cross-section files
        
        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity will only be loaded
            if its molecule is in this list. Mostly used by the 
            :func:`__getitem__` for filtering

        """ 
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
        Main function to use when loading molecular opacities. Handles both 
        cross sections and paths. Handles lists of either so lists of 
        :class:`~taurex.opacity.opacity.Opacity` objects or lists of paths can be used
        to load multiple files/objects


        Parameters
        ----------
        opacities : :class:`~taurex.opacity.opacity.Opacity` or :obj:`list` of :class:`~taurex.opacity.opacity.Opacity` , optional
            Object(s) to include in cache
        
        opacity_path : str or :obj:`list` of str, optional
            search path(s) to look for molecular opacities
        
        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity will only be loaded
            if its molecule is in this list. Mostly used by the 
            :func:`__getitem__` for filtering

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
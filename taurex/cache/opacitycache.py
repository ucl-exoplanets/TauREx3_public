"""
Contains caching class for Molecular cross section files
"""

from .singleton import Singleton
from taurex.log import Logger
from .globalcache import GlobalCache

from taurex.core import Singleton
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

    To get the cross-section object for a particular molecule use the square bracket operator:

    >>> opt['H2O']
    <taurex.opacity.pickleopacity.PickleOpacity at 0x107a60be0>

    This returns a :class:`~taurex.opacity.opacity.Opacity` object for you to compute H2O cross sections from.
    When called for the first time, a directory search is performed and, if found, the appropriate cross-section is loaded. Subsequent calls will immediately
    return the already loaded object:

    >>> h2o_a = opt['H2O']
    >>> h2o_b = opt['H2O']
    >>> h2o_a == h2o_b
    True

    If you have any plugins that include new opacity formats, the cache
    will automatically detect them. 

    

    Lastly you can manually add an opacity directly for a molecule
    into the cache:

    >>> new_h2o = MyNewOpacityFormat()
    >>> new_h2o.molecule
    H2O
    >>> opt.add_opacity(new_h2o)
    >>> opt['H2O']
    <MyNewOpacityFormat at 0x107a60be0>

    


    Now TauREx3 will use it instead in all calculations!

    """
    def init(self):
        self.opacity_dict = {}
        self._opacity_path = None
        self.log = Logger('OpacityCache')
        self._force_active = []
    
    def set_opacity_path(self, opacity_path):
        """
        Set the path(s) that will be searched for opacities.
        Opacities in this path must be of supported types:

            - HDF5 opacities
            - ``.pickle`` opacities
            - ExoTransmit opacities.

        Parameters
        ----------

        opacity_path : str or :obj:`list` of str, optional
            search path(s) to look for molecular opacities

        """

        import os

        GlobalCache()['xsec_path'] = opacity_path

        if not os.path.isdir(opacity_path):
            self.log.error('PATH: %s does not exist!!!', opacity_path)
            raise NotADirectoryError
        self.log.debug('Path set to %s', opacity_path)

    def enable_radis(self, enable):
        """
        Enables/Disables use of RADIS to fill in missing molecules
        using HITRAN.

        .. warning::
            This is extremely unstable and crashes frequently.
            It is also very slow as it requires
            the computation of the Voigt profile for every temperature.
            We recommend leaving it as False unless necessary.

        Parameters
        ----------
        enable: bool
            Whether to enable RADIS functionality (default = False)

        """

        GlobalCache()['enable_radis'] = enable

    def set_radis_wavenumber(self, wn_start, wn_end, wn_points):
        GlobalCache()['radius_grid'] = wn_start, wn_end, wn_points

        self.clear_cache()
    
    def set_memory_mode(self, in_memory):
        """
        If using the HDF5 opacities, whether to stream
        opacities from file (slower, less memory) or load
        them into memory (faster, more memory)

        Parameters
        ----------
        in_memory: bool
            Whether HDF5 files should be streamed (False)
            or loaded into memory (True, default)


        """

        GlobalCache()['xsec_in_memory'] = in_memory
        self.clear_cache()

    def force_active(self, molecules):
        """
        Allows some molecules to be forced as active.
        Useful when using other radiative codes to do the calculation

        Parameters
        ----------
        molecules: obj:`list`
            List of molecules

        """
        self._force_active = molecules


    def set_interpolation(self, interpolation_mode):
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
        GlobalCache()['xsec_interpolation'] = interpolation_mode
        self.clear_cache()
    
    

    def __getitem__(self, key):
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
                self.log.error('Opacity for molecule %s could not be loaded', key)
                self.log.error('It could not be found in the local dictionary %s', list(self.opacity_dict.keys()))
                self.log.error('Or paths %s', GlobalCache()['xsec_path'])
                self.log.error('Try loading it manually/ putting it in a path')
                raise Exception('Opacity could not be loaded')

    def add_opacity(self, opacity, molecule_filter=None):
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
            self.log.warning('Opacity with name %s already in opactiy dictionary %s skipping',opacity.moleculeName,self.opacity_dict.keys())
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
        from taurex.parameter.classfactory import ClassFactory
        opacity_klasses = ClassFactory().opacityKlasses

        molecules = []

        for c in opacity_klasses:
            molecules.extend([x[0] for x in c.discover()])
        
        forced = self._force_active or []
        return set(molecules+forced+list(self.opacity_dict.keys()))

    def load_opacity_from_path(self, path, molecule_filter=None):
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
        from taurex.parameter.classfactory import ClassFactory

        cf = ClassFactory()

        opacity_klass_list = sorted(cf.opacityKlasses,
                                    key=lambda x: x.priority()) 

        for c in opacity_klass_list:

            for mol, args in c.discover():
                self.log.debug('Klass: %s %s', mol, args)
                op = None
                if mol in molecule_filter and mol not in self.opacity_dict:
                    if not isinstance(args, (list, tuple,)):
                        args = [args]
                    op = c(*args)

                if op is not None and op.moleculeName not in self.opacity_dict:
                    self.add_opacity(op, molecule_filter=molecule_filter)
                op = None # Ensure garbage collection when run once
    def load_opacity(self, opacities=None, opacity_path=None, molecule_filter=None):
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
            opacity_path = GlobalCache()['xsec_path']

        if opacities is not None:
            if isinstance(opacities, (list,)):
                self.log.debug('Opacity passed is list')
                for opacity in opacities:
                    self.add_opacity(opacity, molecule_filter=molecule_filter)
            elif isinstance(opacities, Opacity):
                self.add_opacity(opacities, molecule_filter=molecule_filter)
            else:
                self.log.error('Unknown type %s passed into opacities, should be a list, single \
                     opacity or None if reading a path', type(opacities))
                raise Exception('Unknown type passed into opacities')
        else:
            self.load_opacity_from_path(opacity_path, molecule_filter=molecule_filter)
            # if isinstance(opacity_path, str):
            #     self.load_opacity_from_path(opacity_path, molecule_filter=molecule_filter)
            # elif isinstance(opacity_path, (list,)):
            #     for path in opacity_path:
            #         self.load_opacity_from_path(path, molecule_filter=molecule_filter)
    
    def clear_cache(self):
        """
        Clears all currently loaded cross-sections
        """
        self.opacity_dict = {}

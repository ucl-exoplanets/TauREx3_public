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

    Lastly if you've got a hot new opacity format, you can try out
    by manually adding it into the cache:

    >>> new_h2o = MyNewOpacityFormat()
    >>> opt.add_opacity(new_h2o)
    >>> opt['H2O]
    <MyNewOpacityFormat at 0x107a60be0>

    Now TauREx3 will use it instead in all calculations!

    """
    def init(self):
        self.opacity_dict = {}
        self._opacity_path = None
        self.log = Logger('OpacityCache')
        self._default_interpolation = 'linear'
        self._memory_mode = True
        self._radis = False
        self._radis_props = (600, 30000, 10000)
    
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
        if not os.path.isdir(opacity_path):
            self.log.error('PATH: %s does not exist!!!', opacity_path)
            raise NotADirectoryError
        self.log.debug('Path set to %s', opacity_path)
        self._opacity_path = opacity_path

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

        self._radis = enable

    def set_radis_wavenumber(self, wn_start, wn_end, wn_points):
        self._radis_props = (wn_start, wn_end, wn_points)

        

        self.clear_cache()
    
    def set_memory_mode(self,in_memory):
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

        self._memory_mode = in_memory
        self.clear_cache()

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
        return
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
        if key in self.opacity_dict:
            return self.opacity_dict[key]
        else:
            #Try a load of the opacity
            self.load_opacity(molecule_filter=[key])
            #If we have it after a load then good job boys
            if key in self.opacity_dict:
                return self.opacity_dict[key]
            else:
                try:
                    if self._radis:
                        return self.create_radis_opacity(key,molecule_filter=[key])
                    else:
                        raise Exception
                except Exception as e:
                    self.log.error('EXception thrown %s',e)
                    #Otherwise throw an error
                    self.log.error('Opacity for molecule %s could not be loaded',key)
                    self.log.error('It could not be found in the local dictionary %s',list(self.opacity_dict.keys()))
                    self.log.error('Or paths %s',self._opacity_path)
                    self.log.error('Try loading it manually/ putting it in a path')
                    raise Exception('Opacity could notn be loaded')

    def create_radis_opacity(self,molecule,molecule_filter=None):
        from taurex.opacity.radisopacity import RadisHITRANOpacity
        if molecule not in self.opacity_dict:
            self.log.info('Creating Opacity from RADIS+HITRAN')
            wn_start,wn_end,wn_points = self._radis_props
            radis = RadisHITRANOpacity(molecule_name=molecule, wn_start=wn_start,wn_end=wn_end,wn_points=wn_points)
            
            

            self.add_opacity(radis,molecule_filter=molecule_filter)
            return radis
        else:
            self.log.info('Opacity %s already exsits',molecule)


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
            self.log.warning('Opacity with name %s already in opactiy dictionary %s skipping',opacity.moleculeName,self.opacity_dict.keys())
            return
        if molecule_filter is not None:
            if opacity.moleculeName in molecule_filter:
                self.log.info('Loading opacity %s into model',opacity.moleculeName)
                self.opacity_dict[opacity.moleculeName] = opacity       
        else:     
            self.log.info('Loading opacity %s into model',opacity.moleculeName)
            self.opacity_dict[opacity.moleculeName] = opacity   


    def search_hdf5_molecules(self):
        """
        Find molecules with HDF5 opacities in set path

        Returns
        -------
        molecules: :obj`list`
            List of molecules with HDF5 opacities
        
        """
        from glob import glob
        import os    
        from taurex.opacity.hdf5opacity import HDF5Opacity
        glob_path = [os.path.join(self._opacity_path,'*.h5'),os.path.join(self._opacity_path,'*.hdf5')]
        file_list = [f for glist in glob_path for f in glob(glist)]
        
        return [HDF5Opacity(f,interpolation_mode=self._default_interpolation,in_memory=False).moleculeName for f in file_list ]

    def search_pickle_molecules(self):
        """
        Find molecules with ``.pickle`` opacities in set path

        Returns
        -------
        molecules: :obj`list`
            List of molecules with ``.pickle`` opacities
        
        """

        from glob import glob
        import os    
        glob_path = os.path.join(self._opacity_path,'*.pickle')
        file_list = [f for f in glob(glob_path)]
        
        return [pathlib.Path(f).stem.split('.')[0] for f in file_list ]

    def search_exotransmit_molecules(self):
        """
        Find molecules with Exo-Transmit opacities in set path

        Returns
        -------
        molecules: :obj`list`
            List of molecules with ExoTransmit opacities
        
        """

        from glob import glob
        import os    
        glob_path = os.path.join(self._opacity_path,'*.dat')
        file_list = [f for f in glob(glob_path)]
        
        return [pathlib.Path(f).stem[4:] for f in file_list ]

    def search_radis_molecules(self):
        """
        Searches for molecules in HITRAN

        Returns
        -------
        molecules: :obj`list`
            List of molecules available in HITRAN, if radis is enabled,
            otherwise an empty list

        """
        trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     
            '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
            '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl',    '23':'HCN',   '24':'CH3Cl',
            '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
            '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4',  '40':'CH3Br',
            '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N',   '46':'CS',   '47':'SO3'}
        if self._radis:
            return list(trans.values())
        else:
            return []
    def find_list_of_molecules(self):
        from glob import glob
        import os
        from taurex.opacity import PickleOpacity
        pickles = []
        hedef = []
        exo = []
        if self._opacity_path is not None:
        
            pickles = self.search_pickle_molecules()
            hedef = self.search_hdf5_molecules()
            exo = self.search_exotransmit_molecules()
        return list(set(pickles+hedef+exo+self.search_radis_molecules()))
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
        from taurex.opacity.hdf5opacity import HDF5Opacity
        from taurex.opacity.exotransmit import ExoTransmitOpacity
        glob_path = [os.path.join(path,'*.h5'),os.path.join(path,'*.hdf5'),os.path.join(path,'*.pickle'),os.path.join(path,'*.dat')]
    
        file_list = [f for glist in glob_path for f in glob(glist)]
        self.log.debug('File list %s',file_list)
        for files in file_list:
            op = None
            if files.lower().endswith(('.hdf5', '.h5')):
                op = HDF5Opacity(files,interpolation_mode=self._default_interpolation,in_memory=False)
                
                if molecule_filter is not None:
                        if not op.moleculeName in molecule_filter:
                            continue
                if op.moleculeName in self.opacity_dict.keys():
                    continue
                del op

                op = HDF5Opacity(files,interpolation_mode=self._default_interpolation,in_memory=self._memory_mode)

            elif files.endswith('pickle'):
                splits = pathlib.Path(files).stem.split('.')
                if molecule_filter is not None:
                        if not splits[0] in molecule_filter:
                            continue
                if splits[0] in self.opacity_dict.keys():
                    continue
                op = PickleOpacity(files,interpolation_mode=self._default_interpolation)
                op._molecule_name = splits[0]
            elif files.endswith('dat'):
                mol_name = pathlib.Path(files).stem[4:]
                if molecule_filter is not None:
                        if not mol_name in molecule_filter:
                            continue
                if mol_name in self.opacity_dict.keys():
                    continue
                op = ExoTransmitOpacity(files,interpolation_mode=self._default_interpolation)
            if op is not None:
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
"""
Contains caching class for Molecular cross section files
"""

from .singleton import Singleton
from taurex.log import Logger
import pathlib
from . import GlobalCache

class KTableCache(Singleton):
    """
    Implements a lazy load of opacities. A singleton that
    loads and caches xsections as they are needed. Calling

    Now TauREx3 will use it instead in all calculations!

    """
    def init(self):
        self.opacity_dict = {}
        self._opacity_path = GlobalCache()['ktable_path']
        self.log = Logger('KTableCache')
        self._default_interpolation = 'linear'
        self._memory_mode = True

    def set_ktable_path(self, opacity_path):
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
        
        return [pathlib.Path(f).stem.split('.')[0].split('_')[0]  for f in file_list ]


    def find_list_of_molecules(self):
        from glob import glob
        import os
        from taurex.opacity import PickleOpacity
        pickles = []
 
        if self._opacity_path is not None:
        
            pickles = self.search_pickle_molecules()

        return list(set(pickles))

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
        from taurex.opacity.ktables.picklektable import PickleKTable

        glob_path = [os.path.join(path,'*.h5'),os.path.join(path,'*.hdf5'),os.path.join(path,'*.pickle'),os.path.join(path,'*.dat')]
    
        file_list = [f for glist in glob_path for f in glob(glist)]
        self.log.debug('File list %s',file_list)
        for files in file_list:
            op = None
            if files.endswith('pickle'):
                splits = pathlib.Path(files).stem.split('.')[0].split('_')
                if molecule_filter is not None:
                        if not splits[0] in molecule_filter:
                            continue
                if splits[0] in self.opacity_dict.keys():
                    continue
                op = PickleKTable(files)
                op._molecule_name = splits[0]
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
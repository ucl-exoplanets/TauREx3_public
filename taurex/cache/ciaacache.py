"""
Contains caching class for Collisionally Induced Absorption files
"""

from .singleton import Singleton
from taurex.log import Logger


class CIACache(Singleton):
    """
    Implements a lazy load of collisionally induced absorpiton cross-sections
    Supports pickle files and HITRAN cia files. Functionally behaves the same
    as :class:`~taurex.cache.opacitycache.OpacityCache` except the keys are
    now cia pairs
    e.g:

    >>> CIACache()['H2-H2']
    <taurex.cia.picklecia.PickleCIA at 0x107a60be0>

    Pickle ``.db`` and HITRAN ``.cia`` files are supported and automatically
    loaded. with priority given to ``.db`` files



    """
    def init(self):
        self.cia_dict = {}
        self._cia_path = None
        self.log = Logger('CIACache')

    def set_cia_path(self, cia_path):
        """

        Sets the path to search for CIA files

        Parameters
        ----------
        cia_path : str or :obj:`list` of str
            Either a single path or a list of paths that contain CIA files

        """
        self._cia_path = cia_path

    def __getitem__(self, key):
        """
        For a CIA pair, load from the set path and return the
        relevant :class:`~taurex.cia.cia.CIA` object

        Parameter
        ---------
        key : str
            cia pair name

        Returns
        -------
        :class:`~taurex.cia.picklecia.PickleCIA` or :class:`~taurex.cia.hitrancia.HitranCIA`
            Desire CIA object, format depends on what is found
            in the path set by :func:`set_cia_path`

        Raise
        -----
        Exception
            If desired CIA pair could not be loaded

        """

        if key in self.cia_dict:
            return self.cia_dict[key]
        else:
            # Try a load of the cia
            self.load_cia(pair_filter=[key])
            # If we have it after a load then good job boys
            if key in self.cia_dict:
                return self.cia_dict[key]
            else:
                # Otherwise throw an error
                self.log.error('CIA for pair %s could not be loaded', key)
                self.log.error('It could not be found in the local dictionary '
                               ' %s', list(self.cia_dict.keys()))
                self.log.error('Or paths %s', self._cia_path)
                self.log.error('Try loading it manually/ putting it in a path')
                raise Exception('cia could notn be loaded')

    def add_cia(self, cia, pair_filter=None):
        """

        Adds a :class:`~taurex.cia.cia.CIA` object to the cache to then be
        used by Taurex 3

        Parameters
        ----------
        cia : :class:`~taurex.cia.cia.CIA`
            CIA object to add to the cache

        pair_filter : :obj:`list` of str , optional
            If provided, the cia object will only be included
            if its pairname is in the list. Mostly used by the
            :func:`__getitem__` for filtering

        """

        self.log.info('Loading cia %s into model', cia.pairName)
        if cia.pairName in self.cia_dict:
            self.log.error('cia with name %s already in '
                           'opactiy dictionary %s', cia.pairName,
                           self.cia_dict.keys())

            raise Exception('cia for molecule %s already exists')

        if pair_filter is not None:
            if cia.pairName in pair_filter:
                self.log.info('Loading cia %s into model', cia.pairName)
                self.cia_dict[cia.pairName] = cia
        self.cia_dict[cia.pairName] = cia

    def load_cia_from_path(self, path, pair_filter=None):
        """
        Searches path for CIA files, creates and loads them into the cache
        ``.db`` will be loaded as :class:`~taurex.cia.picklecia.PickleCIA` and
        ``.cia`` files will be loaded as
        :class:`~taurex.cia.hitrancia.HitranCIA`

        Parameters
        ----------
        path : str
            Path to search for CIA files

        pair_filter : :obj:`list` of str , optional
            If provided, the cia will only be loaded
            if its pairname is in the list. Mostly used by the
            :func:`__getitem__` for filtering

        """
        from glob import glob
        from pathlib import Path
        import os
        from taurex.cia import PickleCIA

        # Find .db files
        glob_path = os.path.join(path, '*.db')

        file_list = glob(glob_path)
        self.log.debug('Glob list: %s', glob_path)
        self.log.debug('File list FOR CIA %s', file_list)

        for files in file_list:
            pairname = Path(files).stem.split('_')[0]

            self.log.debug('pairname found %s', pairname)

            if pair_filter is not None:
                if pairname not in pair_filter:
                    continue
            op = PickleCIA(files, pairname)
            self.add_cia(op)

        # Find .cia files
        glob_path = os.path.join(path, '*.cia')

        file_list = glob(glob_path)
        self.log.debug('File list %s', file_list)

        for files in file_list:
            from taurex.cia import HitranCIA
            pairname = Path(files).stem.split('_')[0]

            if pair_filter is not None:
                if pairname not in pair_filter:
                    continue
            op = HitranCIA(files)
            self.add_cia(op)

    def load_cia(self, cia_xsec=None, cia_path=None, pair_filter=None):
        """
        Main function to use when loading CIA files. Handles both
        cross sections and paths. Handles lists of either so lists of
        :class:`~taurex.cia.cia.CIA` objects or lists of paths can be used
        to load multiple files/objects


        Parameters
        ----------
        cia_xsec : :class:`~taurex.cia.cia.CIA` or :obj:`list` of :class:`~taurex.cia.cia.CIA` , optional
            Object(s) to include in cache

        cia_path : str or :obj:`list` of str, optional
            search path(s) to look for cias

        pair_filter : :obj:`list` of str , optional
            If provided, the cia will only be loaded
            if its pair name is in this list. Mostly used by the
            :func:`__getitem__` for filtering

        """

        from taurex.cia import CIA
        if cia_path is None:
            cia_path = self._cia_path

        self.log.debug('CIA XSEC, CIA_PATH %s %s', cia_xsec, cia_path)
        if cia_xsec is not None:
            if isinstance(cia_xsec, (list,)):
                self.log.debug('cia passed is list')
                for xsec in cia_xsec:
                    self.add_cia(xsec, pair_filter=pair_filter)
            elif isinstance(cia_xsec, CIA):
                self.add_cia(cia_xsec, pair_filter=pair_filter)
            else:
                self.log.error('Unknown type %s passed into cia, should be a list, single \
                     cia or None if reading a path', type(xsec))
                raise Exception('Unknown type passed into cia')
        if cia_path is not None:

            if isinstance(cia_path, str):
                self.load_cia_from_path(cia_path, pair_filter=pair_filter)
            elif isinstance(cia_path, (list,)):
                for path in cia_path:
                    self.load_cia_from_path(path, pair_filter=pair_filter)

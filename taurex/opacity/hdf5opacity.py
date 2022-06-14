from taurex.cache.globalcache import GlobalCache
from taurex.data.citation import recurse_bibtex, unique_citations_only
from .interpolateopacity import InterpolatingOpacity
import pickle
import numpy as np
import pathlib
from taurex.mpi import allocate_as_shared
from astropy.units import UnitConversionError


class HDF5Opacity(InterpolatingOpacity):
    """
    This is the base class for computing opactities

    """

    @classmethod
    def priority(cls):
        return 5

    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        from taurex.util.util import sanitize_molecule_string

        path = GlobalCache()['xsec_path']
        if path is None:
            return []
        path = [os.path.join(path, '*.h5'), os.path.join(path, '*.hdf5')]
        file_list = [f for glist in path for f in glob.glob(glist)]

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'
        mem = GlobalCache()['xsec_in_memory'] or True

        for f in file_list:
            op = HDF5Opacity(f, interpolation_mode='linear', in_memory=False)
            mol_name = op.moleculeName
            discovery.append((mol_name, [f, interp, mem]))
            # op._spec_dict.close()
            del op

        return discovery

    def __init__(self, filename, interpolation_mode='exp', in_memory=False):
        super().__init__(
            'HDF5Opacity:{}'.format(pathlib.Path(filename).stem[:10]),
            interpolation_mode=interpolation_mode,
        )

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self.in_memory = in_memory
        self._molecular_citation = []
        self._load_hdf_file(filename)

    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def xsecGrid(self):
        return self._xsec_grid

    def _load_hdf_file(self, filename):
        from taurex.data.citation import doi_to_bibtex
        import h5py
        import astropy.units as u
        # Load the pickle file
        self.debug('Loading opacity from {}'.format(filename))

        self._spec_dict = h5py.File(filename, 'r')

        self._wavenumber_grid = self._spec_dict['bin_edges'][:]

        self._temperature_grid = self._spec_dict['t'][:]  # *t_conversion

        pressure_units = self._spec_dict['p'].attrs['units']
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except UnitConversionError:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)

        self._pressure_grid = self._spec_dict['p'][:]*p_conversion

        if self.in_memory:
            self._xsec_grid = allocate_as_shared(
                self._spec_dict['xsecarr'][...], logger=self)
        else:
            self._xsec_grid = self._spec_dict['xsecarr']

        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict['mol_name'][()]

        if isinstance(self._molecule_name, np.ndarray):
            self._molecule_name = self._molecule_name[0]

        try:
            self._molecule_name = self._molecule_name.decode()
        except (UnicodeDecodeError, AttributeError,):
            pass

        from taurex.util.util import ensure_string_utf8

        self._molecule_name = ensure_string_utf8(self._molecule_name)

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()

        try:
            doi = self._spec_dict['DOI'][()]
            if isinstance(doi, np.ndarray):
                doi = doi[0]


            molecular_citation = ensure_string_utf8(
                self._spec_dict['DOI'][()][0])
            new_bib = None
            if not GlobalCache()['xsec_disable_doi']:
                new_bib = doi_to_bibtex(molecular_citation)
            
            self._molecular_citation = [new_bib or molecular_citation]

        except KeyError:
            self._molecular_citation = []

        if self.in_memory:
            self._spec_dict.close()

    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid

    @property
    def pressureGrid(self):
        return self._pressure_grid

    @property
    def resolution(self):
        return self._resolution

    def citations(self):
        from pybtex.database import Entry
        citations = super().citations()
        opacities = []

        for o in self.opacityCitation():
            try:
                e = Entry.from_string(o, 'bibtex')
            except IndexError:
                e = o
            opacities.append(e)
        
        citations = citations + opacities
        return unique_citations_only(citations)


    def opacityCitation(self):
        return self._molecular_citation

    BIBTEX_ENTRIES = [
        """
    @ARTICLE{2021A&A...646A..21C,
        author = {{Chubb}, Katy L. and {Rocchetto}, Marco and {Yurchenko}, Sergei N. and {Min}, Michiel and {Waldmann}, Ingo and {Barstow}, Joanna K. and {Molli{\`e}re}, Paul and {Al-Refaie}, Ahmed F. and {Phillips}, Mark W. and {Tennyson}, Jonathan},
            title = "{The ExoMolOP database: Cross sections and k-tables for molecules of interest in high-temperature exoplanet atmospheres}",
        journal = {Astronomy and Astrophysics},
        keywords = {molecular data, opacity, radiative transfer, planets and satellites: atmospheres, planets and satellites: gaseous planets, infrared: planetary systems, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
            year = 2021,
            month = feb,
        volume = {646},
            eid = {A21},
            pages = {A21},
            doi = {10.1051/0004-6361/202038350},
    archivePrefix = {arXiv},
        eprint = {2009.00687},
    primaryClass = {astro-ph.EP},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...646A..21C},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

        """

    ]

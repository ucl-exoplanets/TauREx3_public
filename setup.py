#!/usr/bin/env python
import setuptools
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy.distutils import log
import re, os

packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex', ]


requires = []


install_requires = ['numpy',
                    'cython',
                    'configobj',
                    'scipy',
                    'numba',
                    'astropy',
                    'numexpr',
                    'numpy',
                    'nestle',
                    'h5py',
                    'tabulate', ]

console_scripts = ['taurex=taurex.taurex:main',
                   'taurex-plot=taurex.plot.plotter:main [Plot]']



def build_ace(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    ace_sources = ['taurex/external/ace.pyf',
    'src/ACE/Md_Types_Numeriques.f90',
                'src/ACE/Md_Constantes.f90',
                'src/ACE/Md_numerical_recipes.f90',
                'src/ACE/Md_Utilitaires.f90','src/ACE/Md_ACE.f90']

    data_files = ('taurex/external/ACE', ['src/ACE/Data/NASA.therm', 'src/ACE/Data/composes.dat'])
    ext = Extension(name='taurex.external.ace', sources=ace_sources)

    return ext, data_files

def build_bhmie():
    return Extension("taurex.external.mie",  
                        sources=["taurex/external/bh_mie.pyx",
                                "src/MIE/bhmie_lib.c",
                                "src/MIE/complex.c",
                                "src/MIE/nrutil.c"
                                ],
                        #extra_compile_args = [],
                        extra_compile_args=["-I./src/MIE/"],
                        language="c")

def _have_fortran_compiler():
    from numpy.distutils.fcompiler import available_fcompilers_for_platform, \
                                            new_fcompiler, \
                                            DistutilsModuleError, \
                                            CompilerNotFound
    from numpy.distutils import customized_fcompiler

    log.info('---------Detecting FORTRAN compilers-------')
    try:
        c = customized_fcompiler()
        v = c.get_version()
        return True
    except (DistutilsModuleError, CompilerNotFound, AttributeError) as e:
        return False

def _have_c_compiler():
    from distutils.errors import DistutilsExecError, DistutilsModuleError, \
                             DistutilsPlatformError, CompileError
    from numpy.distutils import customized_ccompiler
    log.info('---------Detecting C compilers-------')
    try:
        c = customized_ccompiler()
        v = c.get_version()
        return True
    except (DistutilsModuleError, CompileError, AttributeError) as e:
        return False

def create_extensions():
    try:
        from Cython.Build import cythonize
    except ImportError:
        log.warn('Could not import cython, ACE chemistry')
        log.warn('and BH Mie will not be installed')
        return [], []

    extensions = []
    data_files = []
    if _have_fortran_compiler():
        log.info('Detected FORTRAN compiler')
        log.info('ACE chemistry will be installed')

        ext, dat = build_ace()
        extensions.append(ext)
        data_files.append(dat)
    else:
        log.warn('No suitable FORTRAN compiler')
        log.warn('ACE chemistry will not be installed')

    if _have_c_compiler():
        log.info('Detected C compiler')
        log.info('BH Mie will be installed')
        extensions.append(build_bhmie())
    else:
        log.warn('No suitable C compiler')
        log.warn('BH Mie will not be installed')   

    if len(extensions) > 0:
        extensions = cythonize(extensions, language_level=3)

    return extensions, data_files


extensions, data_files = create_extensions()

entry_points = {'console_scripts': console_scripts, }

classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Environment :: Win32 (MS Windows)',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]

# Handle versioning
version = '3.0.3-alpha'

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    setup(name='taurex',
        author='Ahmed Faris Al-Refaie',
        author_email='ahmed.al-refaie.12@ucl.ac.uk',
        license="BSD",
        version=version,
        description='TauREx 3 retrieval framework',
        classifiers=classifiers,
        packages=packages,
        long_description=long_description,
        url='https://github.com/ucl-exoplanets/TauREx3_public/',
        long_description_content_type="text/markdown",
        keywords = ['exoplanet','retrieval','taurex','taurex3','atmosphere','atmospheric'],
        include_package_data=True,
        entry_points=entry_points,
        provides=provides,
        requires=requires,
        install_requires=install_requires,
        extras_require={
            'Plot':  ["matplotlib"], },
        data_files=data_files,
        ext_modules=extensions
        )
except ext_errors as ex:
    print(str(ex))
    print("The C and/or FORTRAN extension could not be compiled")
    setup(name='taurex',
        author='Ahmed Faris Al-Refaie',
        author_email='ahmed.al-refaie.12@ucl.ac.uk',
        license="BSD",
        version=version,
        description='TauREx 3 retrieval framework',
        classifiers=classifiers,
        packages=packages,
        long_description=long_description,
        url='https://github.com/ucl-exoplanets/TauREx3_public/',
        long_description_content_type="text/markdown",
        keywords = ['exoplanet','retrieval','taurex','taurex3','atmosphere','atmospheric'],
        include_package_data=True,
        entry_points=entry_points,
        provides=provides,
        requires=requires,
        install_requires=install_requires,
        extras_require={
            'Plot':  ["matplotlib"], },
        data_files=data_files,
        )
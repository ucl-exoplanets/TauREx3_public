#!/usr/bin/env python


import os
import imp
import setuptools
from setuptools import Distribution
from setuptools import setup, find_packages
from setuptools.command.install import install
from numpy.distutils.core import setup
#from distutils.core import setup
from distutils.command.install_scripts import install_scripts
from distutils import log
from numpy.distutils.core import Extension
from Cython.Build import cythonize



packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex',]


requires = []


install_requires = ['numpy','cython',
        'configobj',
        'scipy',
        'numba',
        'astropy',
        'numexpr',
        'numpy',
        'nestle',
        'h5py',
        'tabulate',]



console_scripts = ['taurex=taurex.taurex:main','taurex-plot=taurex.plot.plotter:main [Plot]']




extensions = []
data_files = []

def build_ace(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    ace_sources = ['taurex/external/ace.pyf',
    'src/ACE/Md_Types_Numeriques.f90',
                'src/ACE/Md_Constantes.f90',
                'src/ACE/Md_numerical_recipes.f90',
                'src/ACE/Md_Utilitaires.f90','src/ACE/Md_ACE.f90']

    data_files = ('taurex/external/ACE', ['src/ACE/Data/NASA.therm', 'src/ACE/Data/composes.dat'])
    ext = Extension(name='taurex.external.ace', sources=ace_sources)

    return ext,data_files

def build_bhmie():
    return Extension("taurex.external.mie",  # indicate where it should be available !
                        sources=["taurex/external/bh_mie.pyx",
                                "src/MIE/bhmie_lib.c",
                                "src/MIE/complex.c",
                                "src/MIE/nrutil.c"
                                ],
                        #extra_compile_args = [],
                        extra_compile_args=["-I./src/MIE/"],
                        language="c")


ext,dat = build_ace()

extensions.append(ext)
data_files.append(dat)
extensions.append(build_bhmie())

extensions = cythonize(extensions, language_level = 3)

entry_points = {'console_scripts': console_scripts,}


classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Environment :: No Input/Output (Daemon)',
    'Environment :: Win32 (MS Windows)',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]


setup(name='taurex',
      author='Foo',
      author_email='bar',
      maintainer='Foo',
      version='3.0',
      description='Bar',
      classifiers=classifiers,
      packages=packages,
      include_package_data=True,
      entry_points=entry_points,
      provides=provides,
      requires=requires,
      install_requires=install_requires,
      extras_require={
        'Plot':  ["matplotlib"],},
      data_files=data_files,
      ext_modules=extensions
      )




# Build the extensions
# --------------------
# The setup steps have been splitted here, because f2py uses a modified version of the setup.
# If only cython or only f2py is used, the setup file can of course be one.

def build_mie():

    clib = Extension("taurex.external.mie",  # indicate where it should be available !
                        sources=["taurex/external/bh_mie.pyx",
                                "src/MIE/bhmie_lib.c",
                                "src/MIE/complex.c",
                                "src/MIE/nrutil.c"
                                ],
                        #extra_compile_args = [],
                        extra_compile_args=["-O3","-I./src/MIE/"],
                        language="c")


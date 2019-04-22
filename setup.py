#!/usr/bin/env python


import os
import imp
from setuptools import Distribution
from setuptools import setup, find_packages
from setuptools.command.install import install
from numpy.distutils.core import setup
#from distutils.core import setup
from distutils.command.install_scripts import install_scripts
from distutils import log
from numpy.distutils.core import Extension



packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex',]


requires = []

install_requires = []



console_scripts = []


def return_major_minor_python():

    import sys

    return str(sys.version_info[0])+"."+str(sys.version_info[1])


def return_include_dir():
    from distutils.util import get_platform
    return get_platform()+'-'+return_major_minor_python()


def ext_configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration()

    #config.add_data_dir('tests')

    ace_src = ['src/ACE/Md_Types_Numeriques.f90',
                'src/ACE/Md_Constantes.f90',
                'src/ACE/Md_numerical_recipes.f90',
                'src/ACE/Md_Utilitaires.f90',
                ]
    config.add_library('ACE', sources=ace_src)


    sources = ['src/ACE/Md_ACE.f90']


    config.add_extension('taurex.external.ace',
        sources=sources,
        libraries=['ace'],
        include_dirs=['build/temp.{}'.format(return_include_dir())],
        depends=(ace_src))
    return config


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

print(ext_configuration(parent_package='taurex').todict())

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
      **ext_configuration(parent_package='taurex').todict())
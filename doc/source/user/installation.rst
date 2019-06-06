

.. _installation:



============
Installation
============

TauREx3 only works with Python 3. If you need to use Python 2.7 consider using TauREx2_.

.. _TauREx2: https://github.com/ucl-exoplanets/TauREx_public

Prerequisites
~~~~~~~~~~~~~

- numpy_

    - Specifically ``numpy.f2py`` to compile Fortran extensions

- cython_

    - Required to compile C++ extensions

- Fortran and C++ compilers

    - For Windows this can be easily achieved using Anaconda3 and doing ``conda install libpython m2w64-toolchain``

All other prerequisites are downloaded and installed automatically.

Installing from git source directly (platform-independent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can clone TauREx3 from our main git repository::

    git clone https://github.com/ucl-exoplanets/TauREx3.git

Move into the TauREx3 folder::

    cd TauREx3

Then, just do::

    pip install -e .

To test for correct setup you can do::

    python -c "import taurex"

If no errors appeared then it was successfuly installed. Additionally the ``taurex`` program 
should now be available in the command line::

    taurex

To build documentation do::
    
    python setup.py build_sphinx

The output can be found in the ``doc/build`` directory



Dependencies
------------

numpy_ and cython_ are the bare minimum required to install TauREx3.
Additionally these packages are also download and installed during setup:

- numba_
- numexpr_
- configobj_ for parsing input files
- nestle_ for basic retrieval
- h5py_ for output

TauREx3 also includes 'extras' that can be enabled by installing 
additional dependancies:

- Lightcurve modelling requires pylightcurve_

- Plotting using ``taurex --plot`` requires matplotlib_

- Retrieval using Multinest_ requires pymultinest_

- Retrieval using PolyChord_ requires pypolychord_

    - The dynamic version requires dypolychord_ as well




.. _numpy: http://numpy.org/
.. _cython: https://cython.org/
.. _configobj: https://pypi.org/project/configobj/
.. _numba: https://numba.pydata.org/
.. _numexpr: https://github.com/pydata/numexpr
.. _nestle: https://github.com/kbarbary/nestle
.. _h5py: https://www.h5py.org/
.. _pylightcurve: https://pypi.org/project/pylightcurve/
.. _matplotlib: https://matplotlib.org/
.. _Multinest: https://github.com/JohannesBuchner/MultiNest
.. _pymultinest: https://github.com/JohannesBuchner/PyMultiNest
.. _PolyChord: https://polychord.io/
.. _pypolychord: https://pypi.org/project/pypolychord/
.. _dypolychord: https://github.com/ejhigson/dyPolyChord/
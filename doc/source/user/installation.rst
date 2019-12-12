

.. _installation:

============
Installation
============

TauREx3 only works with Python 3. If you need to use Python 2.7 consider using TauREx2_.

Prerequisites
~~~~~~~~~~~~~

The only prerequisite is numpy_.
All other prerequisites are downloaded and installed automatically.

Installing from PyPi
~~~~~~~~~~~~~~~~~~~~

Simply do::

    pip install taurex

To test for correct setup you can do::

    python -c "import taurex; print(taurex.__version__)"

Installing from git source directly (platform-independent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can directly get the most cutting-edge release from the repo::

    pip install git+https://github.com/ucl-exoplanets/TauREx3_public.git

You can also clone TauREx3 from our main git repository::

    git clone https://github.com/ucl-exoplanets/TauREx3_public.git

Move into the TauREx3 folder::

    cd TauREx3

Then, just do::

    pip install .

To test for correct setup you can do::

    python -c "import taurex; print(taurex.__version__)"

If no errors appeared then it was successfuly installed.
Additionally the ``taurex`` program should now be available
in the command line::

    taurex

To build documentation do::

    python setup.py build_sphinx

The output can be found in the ``doc/build`` directory

Dependencies
------------

numpy_ is the bare minimum required to install TauREx3.
Additionally these packages are also download and installed during setup:

- numba_
- numexpr_
- configobj_ for parsing input files
- nestle_ for basic retrieval
- h5py_ for output

TauREx3 also includes 'extras' that can be enabled by
installing additional dependancies:

- Lightcurve modelling requires pylightcurve_

- Plotting using ``taurex-plot`` requires matplotlib_

- Retrieval using Multinest_ requires pymultinest_

- Retrieval using PolyChord_ requires pypolychord_

    - The dynamic version requires dypolychord_ as well

Other extras (like equilibrium chemistry)
use external FORTRAN and C++ code. They
require cython_ before installation to compile. Additionally
a FORTRAN compiler and/or C++ compiler must be installed.

.. tip::
    For Windows this can be easily achieved using Anaconda3 and doing ``conda install libpython m2w64-toolchain``


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
.. _TauREx2: https://github.com/ucl-exoplanets/TauREx_public

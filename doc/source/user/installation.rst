

.. _installation:

============
Installation
============

TauREx3 only works with Python 3.5+. If you need to use Python 2.7 consider using TauREx2_.

Installing from PyPi
~~~~~~~~~~~~~~~~~~~~

Simply do::

    pip install taurex

To test for correct setup you can do::

    python -c "import taurex; print(taurex.__version__)"


Additionally, to restore the equilbrium chemistry and BHMie from TauREx 3.0 you can 
run::

    pip install taurex_ace taurex_bhmie




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

As TauREx3 is pure python ,there are no prerequisites.
Additionally these packages are also download and installed during setup:

- numpy_
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

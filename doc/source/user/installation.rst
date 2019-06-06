.. _installation:

============
Installation
============

Prerequisites
~~~~~~~~~~~~~

- numpy
    - Specifically ``numpy.f2py`` to compile Fortran extensions

- cython
    - Required to compile C++ extensions

- Fortran and C++ compilers
    - For Windows this can be easily achieved using Anaconda3 and doing ``conda install libpython m2w64-toolchain``



Installing from git source directly (platform-independent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can clone TauREx3 from our main git repository::

    git clone https://github.com/ucl-exoplanets/TauREx3.git

Move into the TauREx3 folder::

    cd TauREx3

Then, just do::

    pip install -e .

To build documentation do::
    
    python setup.py build_sphinx


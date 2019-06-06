# TauREx3

TauREx3 is the newest version of the TauREx retrieval code that was created by a disturbed man.

## Prerequisites

* numpy
    - Specifically **numpy.f2py** to compile Fortran extensions

* cython
    * Required to compile C++ extensions

* Fortran and C++ compilers
    * For Windows this can be easily achieved using Anaconda3 and doing:
        `conda install libpython m2w64-toolchain`



## Installing

Clone the directory using:

```
git clone https://github.com/ucl-exoplanets/TauREx3.git
```

Move into the TauREx3 folder

```
cd TauREx3
```

Then install

```
pip install -e .
```

To build documentation do

```
python setup.py build_sphinx
```


Try importing taurex:

```
python -c "import taurex"
```

Or running taurex itself

```
taurex
```

If there are no errors then it was successful!
# TauREx3

TauREx3 is the newest version of the TauREx retrieval code.

## Prerequisites

* numpy
    - Specifically **numpy.f2py** to compile Fortran extensions



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
pip install .
```

To build documentation do

```
python setup.py build_sphinx
```


Try importing taurex:

```
python -c "import taurex; print(taurex.__version__)"
```

Or running taurex itself

```
taurex
```

If there are no errors then it was successful!
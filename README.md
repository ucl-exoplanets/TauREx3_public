# TauREx 3
TauREx 3 is the newest version of the TauREx retrieval code.

Documentation can be found [here](https://taurex3-public.readthedocs.io/en/latest/)

Current build: 3.0.3-alpha

## Prerequisites

* numpy



## Installing from PyPi


You can install it by doing

```
pip install taurex
```


## Installing from source


Clone the directory using:

```
git clone https://github.com/ucl-exoplanets/TauREx3_public.git
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
.. _userobservation:

=================
``[Observation]``
=================

This header deals with loading in spectral data
for retrievals or plotting.

--------
Keywords
--------

Only one of these is required. All accept a string path to a file

+-------------------------+---------------------------------------------------------------------+
| Variable                | Data format                                                         |
+-------------------------+---------------------------------------------------------------------+
| ``observed_spectrum``   | ASCII 3/4-column data with format: Wavelength, depth, error, widths |
+-------------------------+---------------------------------------------------------------------+
| ``observed_lightcurve`` | Lightcurve pickle data                                              |
+-------------------------+---------------------------------------------------------------------+
| ``iraclis_spectrum``    | Iraclis output pickle data                                          |
+-------------------------+---------------------------------------------------------------------+
| ``taurex_spectrum``     | TauREX HDF5 output or ``self`` See taurexspectrum_                  |
+-------------------------+---------------------------------------------------------------------+

-------
Example
-------

An example of loading an ascii data-set::

    [Observation]
    observed_spectrum = /path/to/data.dat


.. _taurexspectrum:

TauREx Spectrum
---------------

The ``taurex_spectrum`` has two different modes. The first mode is specifing a filename path of a
a TauREx3 HDF5 output. This output must have been run with some form of instrument function (see :ref:`userinstrument`),
for it to be useable as an observation.
Another is to set ``taurex_spectrum = self``, this will set the current forward model + instrument function
as the observation. This type observation is valid of the fitting procedure making it possible to do *self-retrievals*.



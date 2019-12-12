=================
``[Observation]``
=================

This section deals with loading in spectral data
for retrievals or plotting.

--------
Keywords
--------

Only one of these is required.

+-------------------------+---------------------------------------------------------------------+
| Variable                | Data format                                                         |
+-------------------------+---------------------------------------------------------------------+
| ``observed_spectrum``   | ASCII 3/4-column data with format: Wavelength, depth, error, widths |
+-------------------------+---------------------------------------------------------------------+
| ``observed_lightcurve`` | Lightcurve pickle data                                              |
+-------------------------+---------------------------------------------------------------------+
| ``iraclis_spectrum``    | Iraclis output pickle data                                          |
+-------------------------+---------------------------------------------------------------------+
| ``taurex_spectrum``     | See taurexspectrum_                                                 |
+-------------------------+---------------------------------------------------------------------+



.. _taurexspectrum:

TauREx spectrum
---------------
.. _userbinning:

=============
``[Binning]``
=============

This section deals with the resampling of the forward model.

Binning allows you to change how the forward is sampled. When
only running in forward model mode, it affects the final binned spectrum
stored in the output and the plotting.
It has no effect on retrievals.

The type of binning defined is given by the ``bin_type`` variable:

The available ``bin_type`` are:
    - ``native``
        - Spectra is not resampled
        - Default when no :ref:`userobservation` is given

    - ``observed``
        - Resample to observation grid
        - Default when :ref:`userobservation` is given

    - ``manual``
        - Manually defined resample grid

Manual binning
==============
``bin_type = manual``

When set to manual, you can then define the *start*, *end* and *number of points*
of the grid using one of these keywords:

+-------------------------+----------------------------------------------+
| Variable                | Description                                  |
+-------------------------+----------------------------------------------+
| ``wavelength_grid``     | Equally spaced grid in wavelength (um)       |
+-------------------------+----------------------------------------------+
| ``wavenumber_grid``     | Equally spaced grid in wavenumber (cm-1)     |
+-------------------------+----------------------------------------------+
| ``log_wavelength_grid`` | Equally log-spaced grid in wavelength (um)   |
+-------------------------+----------------------------------------------+
| ``log_wavenumber_grid`` | Equally log-spaced grid in wavenumber (cm-1) |
+-------------------------+----------------------------------------------+

An example, to define an equally spaced wavelength grid at 0.3-5 um::

    [Binning]
    bin_type = manual
    wavelength_grid = 0.3, 5, 300

Or define an equally log spaced wavenumber grid between 400-5000 cm-1::

    [Binning]
    bin_type = manual
    log_wavenumber_grid = 400, 5000, 300    

Alternativly you can instead define it based on the resolution
with the format as *start*, *end*, *resolution*

+-------------------------+----------------------------------------------+
| Variable                | Description                                  |
+-------------------------+----------------------------------------------+
| ``wavelength_res``      | Wavelength grid at resolution (um)           |
+-------------------------+----------------------------------------------+

We can define a grid with 1.1-1.7 um at R=50 resolution::

    [Binning]
    bin_type = manual
    wavelength_res = 1.1, 1.7, 50

Finally there is an optional parameter ``accurate``. When *False*, 
a simpler histogramming method is used to perform the resampling.
When set to ``True`` a more accurate method is used that takes into
account the occupancy of each native sample on the sampling grid.






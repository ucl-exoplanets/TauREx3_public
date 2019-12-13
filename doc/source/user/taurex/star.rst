.. _userstar:

===============
``[Star]``
===============

This header describes the parent star of the exo-planet.
The ``star_type`` informs the type of spectral emission density (SED) used in the emission and direct image forward model.
The ``star_type`` available are:

    - ``blackbody``
        - Star with a blackbody SED
        - Class :class:`~taurex.data.stellar.star.BlackbodyStar`
    - ``phoenix``
        - Uses the PHOENIX_ library for the SED
        - :class:`~taurex.data.stellar.phoenix.PhoenixStar`
    - ``custom``
        - User-provided star model. See :ref:`customtypes`

-------------------------------------

Blackbody
=========
``star_type = blackbody``

Star is considered a blackbody.

--------
Keywords
--------

+------------------+--------------+----------------------------+---------+
| Variable         | Type         | Description                | Default |
+------------------+--------------+----------------------------+---------+
| ``temperature``  | :obj:`float` | Effective temperature in K | 5000    |
+------------------+--------------+----------------------------+---------+
| ``radius``       | :obj:`float` | Radius in solar radius     | 1.0     |
+------------------+--------------+----------------------------+---------+
| ``mass``         | :obj:`float` | Mass in solar mass         | 1.0     |
+------------------+--------------+----------------------------+---------+
| ``distance``     | :obj:`float` | Distance from Earth in pc  | 1.0     |
+------------------+--------------+----------------------------+---------+
| ``metallicity``  | :obj:`float` | Metallicity in solar units | 1.0     |
+------------------+--------------+----------------------------+---------+
| ``magnitudeK``   | :obj:`float` | Magnitude in K-band        | 10.0    |
+------------------+--------------+----------------------------+---------+

--------
Examples
--------

A Sun like star as a black body::

    [Star]
    star_type = blackbody
    radius = 1.0
    temperature = 5800

-----------------------------------

PHOENIX
========
``star_type = phoenix``

Stellar emission spectrum is read from the PHOENIX_ library ``.fits.gz`` files and interpolated to the correct temperature.
Any temperature outside of the range provided by PHOENIX will use a blackbody SED instead.
The ``.fits.gz`` filenames must include the temperature as the first number. TauREx3 splits the filename
in terms of numbers so any text can be included in the beginning of the file name, therefore these are valid::
    lte05600.fits.gz  # 5600 Kelvin
    abunchofothertext-andanother-here05660-0.4_0.5.0.8.fits.gz #5660 Kelvin
    5700-056-034-0434.fits.gz #5700 Kelvin

--------
Keywords
--------

+------------------+--------------+----------------------------+--------------+
| Variable         | Type         | Description                | Default      |
+------------------+--------------+----------------------------+--------------+
| ``phoenix_path`` | :obj:`str`   | Path to ``.fits.gz`` files | **Required** |
+------------------+--------------+----------------------------+--------------+
| ``temperature``  | :obj:`float` | Effective temperature in K | 5000         |
+------------------+--------------+----------------------------+--------------+
| ``radius``       | :obj:`float` | Radius in solar radius     | 1.0          |
+------------------+--------------+----------------------------+--------------+
| ``mass``         | :obj:`float` | Mass in solar mass         | 1.0          |
+------------------+--------------+----------------------------+--------------+
| ``distance``     | :obj:`float` | Distance from Earth in pc  | 1.0          |
+------------------+--------------+----------------------------+--------------+
| ``metallicity``  | :obj:`float` | Metallicity in solar units | 1.0          |
+------------------+--------------+----------------------------+--------------+
| ``magnitudeK``   | :obj:`float` | Magnitude in K-band        | 10.0         |
+------------------+--------------+----------------------------+--------------+

--------
Examples
--------

A Sun like star using PHOENIX spectra::

    [Star]
    star_type = phoenix
    radius = 1.0
    temperature = 5800
    phoenix_path = /mypath/to/fitsfiles/





.. _PHOENIX: https://arxiv.org/abs/1303.5632

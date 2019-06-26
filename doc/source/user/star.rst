.. _star:

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



Blackbody
---------
``star_type = blackbody``

Star is considered a blackbody. Available variables are:
    - ``radius``
        - float
        - Radius in Solar radii
        - Default: ``radius = 1.0``
    - `` temperature``
        - float
        - Temperature in Kelvin
        - Default ``temperature = 5800``

PHOENIX
-------
``star_type = phoenix``

Stellar emission spectrum is read from the PHOENIX_ library and interpolated to the correct temperature.
Any temperature outside of the range provided by PHOENIX will use a blackbody SED instead.
The variables available are:

    - ``radius``
        - float
        - Radius in Solar radii
        - Default: ``radius = 1.0``
    - `` temperature``
        - float
        - Temperature in Kelvin
        - Default ``temperature = 5800``
    - ``phoenix_path``
        - str
        - Path to ``.fmt`` files
        - **Required**

The ``.fmt`` filenames must include the temperature as the first number. TauREx3 splits the filename
in terms of numbers so any text can be included in the beginning of the file name, therefore these are valid::
    lte05600.fmt  # 5600 Kelvin
    abunchofothertext-andanother-here05660-0.4_0.5.0.8.fmt #5660 Kelvin
    5700-056-034-0434.fmt #5700 Kelvin







.. _PHOENIX: <https://arxiv.org/abs/1303.5632>

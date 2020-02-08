.. _userpressure:

===============
``[Pressure]``
===============

The header describes pressure profiles for the atmosphere.
Currently only one type of profile is supported, so ``profile_type=simple`` or ``profile_type=hydrostatic`` must be included.
``profile_type = custom`` is also valid, See :ref:`customtypes` 

:Class: :class:`~taurex.data.profiles.pressure.SimplePressureProfile`

--------
Keywords
--------

+----------------------+--------------+-------------------------+---------+
| Variable             | Type         | Description             | Default |
+----------------------+--------------+-------------------------+---------+
| ``atm_min_pressure`` | :obj:`float` | Pressure in Pa at TOA   | 1e0     |
+----------------------+--------------+-------------------------+---------+
| ``atm_max_pressure`` | :obj:`float` | Pressure in Pa at BOA   | 1e6     |
+----------------------+--------------+-------------------------+---------+
| ``nlayers``          | :obj:`int`   | Number of layers        | 100     |
+----------------------+--------------+-------------------------+---------+


------------------
Fitting Parameters
------------------

.. warning::

    Whilst included of completeness it is generally not a good idea
    to fit these parameters as it can drastically alter the scale of
    the atmosphere.

+----------------------+--------------+-------------------------+
| Variable             | Type         | Description             |
+----------------------+--------------+-------------------------+
| ``atm_min_pressure`` | :obj:`float` | Pressure in Pa at TOA   |
+----------------------+--------------+-------------------------+
| ``atm_max_pressure`` | :obj:`float` | Pressure in Pa at BOA   |
+----------------------+--------------+-------------------------+
| ``nlayers``          | :obj:`int`   | Number of layers        |
+----------------------+--------------+-------------------------+

Examples
--------

A basic pressure profile::

    [Pressure]
    profile_type = simple
    atm_min_pressure = 1e-3
    atm_max_pressure = 1e6
    nlayers = 100

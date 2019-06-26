.. _pressure:

===============
``[Pressure]``
===============

The header describes pressure profiles for the atmosphere.
Currently only one type of profile is supported, so ``profile_type=simple`` must be included.
The :class:`~taures.data.profiles.pressure.pressureprofile.SimplePressureProfile` profile.
The three variables are:
    - ``atm_min_pressure``
        - Minimum pressure of atmosphere in Pascal (Top)
    - ``atm_max_pressure``
        - Maximum pressure of atmosphere in Pascal (Surface)
    - ``nlayers``
        - Number of layers in atmosphere.
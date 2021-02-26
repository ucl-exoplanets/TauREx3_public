.. _userderived:

=============
``[Derive]``
=============

.. versionadded:: 3.1

This section deals with post-processed values from a retrieval.

The format for enabling post-processed values are::

    derived_param:compute = value

Only ``compute`` is available as an option. Setting to ``True`` will
ask TauREx to generate posteriors at the end of a retrieval 
for the parameter using the sample points.

By default, the chemistry mean molecular mass (:math:`\mu`) at the surface is computed.
We can disable this and instead compute the :math:`log(g)` of the planet surface
and average temperature like so::

    [Derive]
    mu:compute = False
    logg:compute = True
    avg_T:compute = True

Refer to the documentation or plugin documentation to find out what derived parameters
are available. You can pass your input file with the ``--fitparam`` option to list
available parameters::

    > taurex -i myinput.par --fitparam

With the derived paramaters listed under ``Available Computable Parameters``::

    -----------------------------------------------
    ------Available Retrieval Parameters-----------
    -----------------------------------------------

    ╒══════════════════╤══════════════════════════════════════════════════════╕
    │ Param Name       │ Short Desc                                           │
    ╞══════════════════╪══════════════════════════════════════════════════════╡
    │ planet_mass      │ Planet mass in Jupiter mass                          │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ planet_radius    │ Planet radius in Jupiter radii                       │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ planet_distance  │ Planet semi major axis from parent star (AU)         │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ planet_sma       │ Planet semi major axis from parent star (AU) (ALIAS) │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ atm_min_pressure │ Minimum pressure of atmosphere (top layer) in Pascal │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ atm_max_pressure │ Maximum pressure of atmosphere (surface) in Pascal   │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ T                │ Isothermal temperature in Kelvin                     │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ H2O              │ H2O constant mix ratio (VMR)                         │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ CH4              │ CH4 constant mix ratio (VMR)                         │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ He_H2            │ He/H2 ratio (volume)                                 │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ clouds_pressure  │ Cloud top pressure in Pascal                         │
    ╘══════════════════╧══════════════════════════════════════════════════════╛




    -----------------------------------------------
    ------Available Computable Parameters----------
    -----------------------------------------------

    ╒══════════════╤════════════════════════════════════════╕
    │ Param Name   │ Short Desc                             │
    ╞══════════════╪════════════════════════════════════════╡
    │ logg         │ Surface gravity (m2/s) in log10        │
    ├──────────────┼────────────────────────────────────────┤
    │ avg_T        │ Average temperature across all layers  │
    ├──────────────┼────────────────────────────────────────┤
    │ mu           │ Mean molecular weight at surface (amu) │
    ├──────────────┼────────────────────────────────────────┤
    │ C_O_ratio    │ C/O ratio (volume)                     │
    ├──────────────┼────────────────────────────────────────┤
    │ He_H_ratio   │ He/H ratio (volume)                    │
    ╘══════════════╧════════════════════════════════════════╛


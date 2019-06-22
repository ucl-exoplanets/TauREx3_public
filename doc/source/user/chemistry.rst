.. _chemistry:

===============
``[Chemistry]``
===============

This header describes the chemical composition of the
atmosphere. The type of model used is defined by the
``chemistry_type`` variable.

The available ``chemistry_types`` are:
    - ``ace``
        - ACE equlibrium chemistry
        - Class: :class:`~taurex.data.profiles.chemistry.acechemistry.ACEChemistry`
        - Alt: ``equilibrium``
    - ``taurex``
        - Custom chemistry
        - Class: :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry`



ACE Equlibrium Chemistry
========================
``chemistry_type = ace``


Equilibrium chemistry using the ACE FORTRAN program.

The variables available are:
    - ``active_gases``
        - List of molecules
        - Defines which molecules are actively absorbing
        - Default: ``active_gases = H2O,CH4``
    - ``metallicity``
        - float
        - Stellar metallicity in solar units
        - Default: ``metallicity = 1.0``
    - ``co_ratio``
        - float
        - C/O ratio
        - Default: ``co_ratio=0.54951``

Taurex Chemistry
===========================
``chemistry_type = taurex``


This chemistry type allows you to define individual
abundance profiles for each molecule.

On its own it has the variables:
    - ``n2_mix_ratio``
        - float
        - Abundance of ``N2``
        - Default: ``n2_mix_ratio = 0``

    - ``he_h2_ratio``
        - float
        - Ratio of ``H2`` to ``He`` to fill remainder of atmosphere with
        - Default: ``he_h2_ratio = 0.17647``

However molecules are defined as *subheaders* with the subheader being the name of the molecule.
Each molecule can be assigned an abundance profile through the ``gas_type`` variable.
For example, to describe a chemical profile with water in constant abundance in the atmosphere 
is simply done like so::

    [Chemistry]
    chemistry_type = taurex
    he_h2_ratio = 0.1524

        [[H2O]]
        gas_type = constant
        mix_ratio = 1e-4

For each molecule, the available ``gas_type`` are:
    - ``constant``
       - Constant abundance profile
       - Class: :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`

    - ``twopoint``
        - Two Point abundance profile
        - Class: :class:`~taurex.data.profiles.chemistry.gas.twopointgas.TwoPointGas`
    
    - ``twolayer``
        - Two layer abundance profile
        - Class: :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`


Constant Profile
----------------
``gas_type = constant``

An abundance profile that is constant with height of the atmosphere

.. figure::  _static/constantgas.png
   :align:   left
   :width: 80%

Variables are:
    - ``mix_ratio``
        - float
        - The abundance for every layer in the atmosphere

Two Point Profile
-----------------
``gas_type = twopoint``

An abundance profile where abundance is defined on the planet surface and top of
the atmosphere and interpolated

.. figure::  _static/twopointgas.png
   :align:   left
   :width: 80%

Variables are:
    - ``mix_ratio_surface``
        - float
        - Abundance on the planet surface
    - ``mix_ratio_top``
        - float
        - Abundance on the top of that atmosphere



Two Layer Profile
-----------------
``gas_type = twolayer``

An abundance profile where abundance is defined on the planet surface and top of
the atmosphere with a pressure point determining the boundary between the layers.
Smoothing is applied.

.. figure::  _static/twolayerabundance.png
   :align:   left
   :width: 80%

Variables are:
    - ``mix_ratio_surface``
        - float
        - Abundance on the planet surface
    - ``mix_ratio_top``
        - float
        - Abundance on the top of that atmosphere
    - ``mix_ratio_P``
        - float
        - Pressure point that seperates the top and surface
    - ``mix_ratio_smoothing``
        - int
        - Smoothing window
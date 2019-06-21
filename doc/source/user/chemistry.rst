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

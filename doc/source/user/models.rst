.. _model:

===========
``[Model]``
===========

This header defines the type of forward model (FM) that will be computed by TauREx3.
There are only four distinct forward ``model_type``:
    - ``transmission``
        - Transmission forward model
    - ``emission``
        - Emission forward model
    - ``directimage``
        - Direct-image forward model
    - ``mars``
        - Mars forward model



Mars
----

.. warning::
    This will be completed once the current Mars branch is finished.



Contributions
-------------

Contributions define what processes in the atmosphere contribute to the optical depth.
These contributions are defined as *subheaders* with the name of the header being the contribution 
to add into the forward model. Any forward model type can be augmented with these contributions.
The available contributions are:
    - ``[[Absorption]]``
        - Adds molecular absorption to the FM
        - Class: :class:`~taurex.contributions.absorption.AbsorptionContribution`
        - No variables
    - ``[[CIA]]``
        - Includes collisionally induced absorption processes in FM
        - Class: :class:`~taurex.contributions.cia.CIAContribution`
        - Variables:
            - ``cia_pairs``
                - list of comma seperated 
                  molecule pairs. e.g ``H2-He``, ``N2-N2``
    - ``[[Rayleigh]]``
        - Added Rayleigh scattering to FM
        - Class: :class:`~taurex.contributions.rayleigh.RayleighContribution`
        - No variables

Examples
--------

Transmission spectrum with molecular absorption and CIA from ``H2-He`` and ``H2-H2``::

    [Model]
    model_type = transmission
        [[Absorption]]

        [[CIA]]
        cia_pairs = H2-He,He-He
    
Emission spectrum with molecular absorption, CIA and Rayleigh scattering::

    [Model]
    model_type = emission
        [[Absorption]]

        [[CIA]]
        cia_pairs = H2-He,He-He  

        [[Rayleigh]]



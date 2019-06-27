.. _planet:

============
``[Planet]``
============

This header is used to define planetary properties. Currently, only ``planet_type = simple``
is supported and must be included.

Variables
---------
    - ``mass``
        - float
        - Mass in Jupiter mass
        - Default: ``mass = 1.0``
    - ``radius``
        - float
        - Radius in Jupiter radius
        - Default: ``radius = 1.0``

Examples
--------

Planet with 1.5 Jupiter mass and 1.2 Jupiter radii::

    [Planet]
    planet_type = simple
    mass = 1.5
    radius = 1.2

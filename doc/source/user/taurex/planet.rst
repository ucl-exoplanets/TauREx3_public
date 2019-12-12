.. _planet:

============
``[Planet]``
============

This header is used to define planetary properties. Currently, only ``planet_type = simple``
is supported and must be included.


--------
Keywords
--------

+---------------------+--------------+--------------------------+---------+
| Variable            | Type         | Description              | Default |
+---------------------+--------------+--------------------------+---------+
| ``planet_mass``     | :obj:`float` | Mass in Jupiter mass     | 1.0     |
+---------------------+--------------+--------------------------+---------+
| ``planet_radius``   | :obj:`float` | Radius in Jupiter radius | 1.0     |
+---------------------+--------------+--------------------------+---------+
| ``planet_distance`` | :obj:`float` | Semi-major-axis in AU    | 1.0     |
+---------------------+--------------+--------------------------+---------+
| ``impact_param``    | :obj:`float` | Impact parameter         | 0.5     |
+---------------------+--------------+--------------------------+---------+
| ``orbital_period``  | :obj:`float` | Orbital period in days   | 2.0     |
+---------------------+--------------+--------------------------+---------+
| ``albedo``          | :obj:`float` | Planetary albedo         | 0.3     |
+---------------------+--------------+--------------------------+---------+
| ``transit_time``    | :obj:`float` | Transit time in seconds  | 3000.0  |
+---------------------+--------------+--------------------------+---------+

------------------
Fitting Parameters
------------------

+---------------------+--------------+--------------------------+
| Parameter           | Type         | Description              |
+---------------------+--------------+--------------------------+
| ``planet_mass``     | :obj:`float` | Mass in Jupiter mass     |
+---------------------+--------------+--------------------------+
| ``planet_radius``   | :obj:`float` | Radius in Jupiter radius |
+---------------------+--------------+--------------------------+
| ``planet_distance`` | :obj:`float` | Semi-major-axis in AU    |
+---------------------+--------------+--------------------------+


Examples
--------

Planet with 1.5 Jupiter mass and 1.2 Jupiter radii::

    [Planet]
    planet_type = simple
    planet_mass = 1.5
    planet_radius = 1.2

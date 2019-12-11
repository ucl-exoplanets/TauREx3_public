.. _global:

============
``[Global]``
============

The global section generally handles settings that affect the whole program.

- ``xsec_path``
    - str or list of str
    - Defines the path(s) that contain molecular cross-sections
    - e.g ``xsec_path = path/to/xsec``

- ``xsec_interpolation``
    - ``exp`` or ``linear``
    - Defines whether to use exponential or linear interpolation for temperature
    - e.g ``xsec_interpolation = exp``


- ``cia_path``
    - str or list of str
    - Defines the path(s) that contain CIA cross-sections
    - e.g ``xsec_path = path/to/xsec``

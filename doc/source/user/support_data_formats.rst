.. _supported_data_formats:

======================
Supported Data Formats
======================


Cross-sections
~~~~~~~~~~~~~~
Supported formats are:

- ``.pickle`` *TauREx2* pickle format
- ``.hdf5``, ``.h5`` New HDF5 format
- ``.dat``,  ExoTransmit_ format

More formats can be included through :ref:`plugins`

.. tip::

    For opacities we recommend using hi-res cross-sections (R>7000)
    from a high temperature linelist. Our recommendation are
    linelists from the ExoMol_ project.

K-Tables
~~~~~~~~

.. versionadded:: 3.1


Supported formats are:

- ``.pickle`` *TauREx2* pickle format
- ``.hdf5``, ``.h5`` petitRADTRANS HDF5 format
- ``.kta``,  NEMESIS format

More formats can be included through :ref:`plugins`


Observation
~~~~~~~~~~~

For observations, the following formats supported
are:

- Text based 3/4-column data
- ``.pickle`` Outputs from Iraclis_

More formats can be included through :ref:`plugins`


Collisionally Induced Absorption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only a few formats are supported

- ``.db`` *TauREx2* CIA pickle files
- ``.cia`` HITRAN_ cia files

.. _HITRAN: https://hitran.org/cia/

.. _ExoTransmit: https://github.com/elizakempton/Exo_Transmit/tree/master/Opac

.. _Iraclis: https://github.com/ucl-exoplanets/Iraclis

.. _ExoMol: http://www.exomol.com
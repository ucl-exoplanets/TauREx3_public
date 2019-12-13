.. _supported_data_formats:

======================
Supported Data Formats
======================


Cross-sections
~~~~~~~~~~~~~~

k-tables are *no longer supported*.
Only molecular cross-sections are supported

Supported formats are:

- ``.pickle`` *Taurex2* pickle format
- ``.hdf5``, ``.h5`` New HDF5 format
- ``.dat``,  ExoTransmit_ format

Observation
~~~~~~~~~~~

For observations, the following formats supported
are:

- Text based 3/4-column data
- ``.pickle`` Outputs from Iraclis_


Collisionally Induced Absorption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only a few formats are supported

- ``.db`` *Taurex2* CIA pickle files
- ``.cia`` HITRAN_ cia files

.. _HITRAN: https://hitran.org/cia/

.. _ExoTransmit: https://github.com/elizakempton/Exo_Transmit/tree/master/Opac

.. _Iraclis: https://github.com/ucl-exoplanets/Iraclis

.. _userinstrument:

================
``[Instrument]``
================

This section deals with passing the forward model through
some form of noise model.

The instrument function in TauREx3 serves to generate
a spectrum and noise from a forward model.

Including a noise model in the TauREx3 input makes the output file capable
of being used as an observation in the retrieval. It is also capable
of fitting itself (See :ref:`taurexspectrum`)

The instrument is defined by the ``instrument`` variable:
    - ``snr``
        - Signal-to-noise ratio instrument
        - Class: :class:`~taurex.instruments.snr.SNR`

    - ``custom``
        - User-type instrument. See :ref:`customtypes`

----------------------

SNR
===
``instrument = snr``
A very basic instrument that generates noise based on the forward model
spectrum and signal-to-noise ratio value. Uses the native spectrum as the grid,
unless a :ref:`manualbinning` is defined in which case that is used as the grid.

--------
Keywords
--------

+-------------------------+--------------+----------------------------------------------+
| Variable                | Type         | Description                                  |
+-------------------------+--------------+----------------------------------------------+
| ``SNR``                 | :obj:`float` | Signal-to-noise ratio                        |
+-------------------------+--------------+----------------------------------------------+
| ``num_observation``     | :obj:`int`   | Number of observations                       |
+-------------------------+--------------+----------------------------------------------+


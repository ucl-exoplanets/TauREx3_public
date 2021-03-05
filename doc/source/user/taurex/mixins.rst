.. _mixin:

======
Mixins
======

.. versionadded:: 3.1

Mixins are lighter components with the sole purpose of giving 
*all* atmospheric components new abilities and features. For the coding
inclined you can see the article `here <wiki_>`_.

``makefree``
============

Works under: ``[Chemistry]``

Adds new molecules or forces specific molecules in a chemical scheme to become fittable.
Molecules will behave like a :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
and will add them as fitting parameters. Molecules can be defined similarly to ``free``.
For example if we load a chemistry from file, normally we cannot retrieve any molecule.
If add the ``makefree`` mixin we can force specific molecules and add in new molecules into the
scheme::

    [Chemistry]
    chemistry_type = makefree+file
    filename = "mychemistryprofile.dat"
    gases = H2O, CH4, CO, CO2

        [CH4]
        gas_type = constant

        [TiO]
        gas_type = constant

    [Fitting]
    CH4:fit = True
    TiO:fit = True

Here, CH4 will has become fittable and we injected TiO into the scheme. What happens is that
each time the chemistry will run it will first run the base scheme and then modify or inject
the molecule into the profile. Afterwhich the mixing profiles are then renormalized to unity.
This can also work for equlibrium schemes, for example using ACE::

    [Chemistry]
    chemistry_type = makefree+ace
    metallicity = 1.0

        [CH4]
        gas_type = constant

        [TiO]
        gas_type = constant

    [Fitting]
    CH4:fit = True
    TiO:fit = True
    metallicity:fit = True

Only the ``free`` chemical scheme does not work as it is redundant.


.. _wiki: https://en.wikipedia.org/wiki/Mixin
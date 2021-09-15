.. _retreivaldev:

====================
Retreival Parameters
====================

Fitting
-------

TauREx 3 employs dynamic discovery of retrieval parameters.
When a new profile/chemistry etc is loaded into a forward model,
they also advertise the parameters that can be retrieved.
The forward model will collect and provide them to the
optimizer which it then uses to perform the sampling.

Classes that inherit from :class:`~taurex.data.fittable.Fittable` are capable
of having retrieval parameters. Classes that inherit from this include:

    - :class:`~taurex.data.profiles.temperature.tprofile.TemperatureProfile`
    - :class:`~taurex.data.profiles.chemistry.chemistry.Chemistry`
    - :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
    - :class:`~taurex.data.profiles.pressure.pressureprofile.PressureProfile`
    - :class:`~taurex.data.stellar.star.Star`
    - :class:`~taurex.data.planet.Planet`
    - :class:`~taurex.model.model.ForwardModel`
    - :class:`~taurex.contributions.contribution.Contribution`

There are two ways of defining, fitting parameters. The simpler decorator method
and programmaticaly


Decorator form
---------------

The decorator :func:`~taurex.data.fittable.fitparam` decorator acts and behaves almost identically to
the ``@property`` python decorator_.




Programmaticaly
---------------

The decorator form is useful for describing parameters that always exist in
a model in the same form. There are cases where parameters may need to be
generated based on some input. One example is the :class:`~taurex.data.profiles.temperature.npoint.NPoint` temperature profile.
Depending on the number of temperature points input by the user the temperature profile will actually generate *new* fitting parameters
for each point (i.e ``T_point1``, ``T_point2`` etc)
Another example are the :class:`~taurex.data.profiles.chemistry.gas.gas.Gas` profiles.
In this case, the fitting parameter names ar




.. _decorator: https://docs.python.org/3/library/functions.html#property
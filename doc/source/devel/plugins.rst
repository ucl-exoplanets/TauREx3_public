.. _buildplugin:

==================
Developing Plugins
==================

Atmospheric retrievals are not an isolated science. We regularly use different codes and models from various fields to better characterise exoplanetary systems. Often repetitive steps are needed to make use of an external code, and frequently, these are difficult to share or distribute to a broader audience.
Plugins are a new feature in TauREx 3.1 that allows developers to simplify the distribution and usage of their profile/models/chemistry etc., to other users for their retrievals.
The plugin system can be used to add the following new components:

    - :class:`~taurex.data.profiles.temperature.tprofile.TemperatureProfile`
    - :class:`~taurex.data.profiles.chemistry.chemistry.Chemistry`
    - :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
    - :class:`~taurex.data.profiles.pressure.pressureprofile.PressureProfile`
    - :class:`~taurex.data.stellar.star.Star`
    - :class:`~taurex.data.planet.Planet`
    - :class:`~taurex.model.model.ForwardModel`
    - :class:`~taurex.contributions.contribution.Contribution`
    - :class:`~taurex.opacity.opacity.Opacity`






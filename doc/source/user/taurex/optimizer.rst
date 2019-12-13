.. _useroptimizer:

===============
``[Optimizer]``
===============

This section deals with the type of optimizer used.

TauREx3 includes a few samplers to perform retrievals. These can be set using the ``optimizer``
keyword:

    - ``nestle``
        - Nestle sampler
        - Class :class:`~taurex.optimizer.nestle.NestleOptimizer`
    - ``multinest``
        - Use the MultiNest sampler
        - Class :class:`~taurex.optimizer.multinest.MultiNestOptimizer`
    - ``polychord``
        - PolyChord Optimizer
        - Class :class:`~taurex.optimizer.polychord.PolyChordOptimizer`
    - ``dypolychord``
        - dyPolyChord optimizer
        - Class :class:`~taurex.optimizer.dypolychord.dyPolyChordOptimizer`
    - ``custom``
        - User-provided star model. See :ref:`customtypes`
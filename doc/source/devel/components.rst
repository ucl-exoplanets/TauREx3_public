==========
Components
==========

.. warning::
    
    This is still under construction as I try to figure out how best to convey this.


Here we present the most basic form of each component. See
basic features under :ref:`basics` and retrieval features under 
:ref:`retrievaldev`


Temperature
===========

The most basic temperature class has this form:

.. code-block:: python

    from taurex.temperature import Temperature
    import numpy as np

    class MyTemperature(Temperature):

        def __init__(self):
            super().__init__(self.__class__.__name__)
        
        def initialize_profile(self, planet=None, nlayers=100,
                            pressure_profile=None):
            self.nlayers = nlayers

            self.myprofile = np.ones(nlayers)*1000.0

        @property
        def profile(self):
            return self.myprofile

:meth:`~taurex.data.profiles.temperature.tprofile.Temperature.__init__`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Used to build the component and only called once. Must include ``super()`` call. 
Decorator fitting parameters are also collected here automatically. 
Use keyword arguments to setup the class and load any necessary files. You can also build new fitting
parameters here as well.
 
:meth:`~taurex.data.profiles.temperature.tprofile.Temperature.initialize_profile`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to initialize and compute the temperature profile.
It is run on each :meth:`~taurex.model.simplemodel.SimpleForwardModel.model` call

Arguments: 
    - ``planet``: :class:`~taurex.data.planet.Planet`
    - ``nlayers``: Number of Layers
    - ``pressure_profile``: ``nlayer`` array of pressures.
        - BOA to TOA
        - Units: :math:`Pa`

:meth:`~taurex.data.profiles.temperature.tprofile.Temperature.profile`

Must be decorated with `@property <decorator_>`_. Must return an array of
same shape as ``pressure_profile`` with units :math:`K`


Chemistry
=========

We recommend using :class:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry`
as a base as it greatly simplifies implementation of active and inactive species.

.. code-block:: python

    from taurex.chemistry import AutoChemistry
    import numpy as np

    class MyChemistry(AutoChemistry):

        def __init__(self):
            super().__init__(self.__class__.__name__)

            # Perform setup here
            
            # Populate gases here
            self.mygases = ['H2', 'He', 'H2O', 'CH4', 'NO', 'H2S','TiO',]

            # Call when gases has been populated
            self.determine_active_inactive()
        
        def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                            pressure_profile=None, altitude_profile=None):

            num_molecules = len(self.gases)


            # We will compute a random profile for each molecule
            self.mixprofile = np.random.rand(num_molecules, nlayers)

            # Make sure each layer sums to unity
            self.mixprofile/= np.sum(self.mixprofile,axis=0)

            # Compute mu profile
            self.compute_mu_profile(nlayers):

        @property
        def gases(self):
            return self.mygases
    
        @property
        def mixProfile(self):
            return self.mixprofile

For chemistry whats important is the the method :meth:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry.determine_active_inactive`
must be called once :meth:`~taurex.data.profiles.chemistry.AutoChemistry.gases` has been populated with the species.

:meth:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry.__init__`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to build the component and only called once. Must include ``super()`` call. 
Decorator fitting parameters are also collected here automatically. 
Use keyword arguments to setup the class and load any necessary files. You can also build new fitting
parameters here as well. We recommend determining your chemical species at this point.
 
:meth:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry.initialize_chemistry`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Used to initialize and compute the chemical model. The :math:`\mu` profile should be computed
as well. It is run on each :meth:`~taurex.model.simplemodel.SimpleForwardModel.model` call

Arguments: 
    - ``nlayers``: Number of Layers
    - ``temperature_profile``: ``nlayer`` array of temperature.
        - BOA to TOA
        - Units: :math:`K`
    - ``pressure_profile``: ``nlayer`` array of pressures.
        - BOA to TOA
        - Units: :math:`Pa`

:meth:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry.gases`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Must be decorated with `@property <decorator_>`_. Must return a list of species
in the chemical model

:meth:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry.mixProfile`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return volume mixing ratios. Must be decorated with `@property <decorator_>`_. Must return an array of shape
(number of species, number of layers). The ordering of species must be 1:1 with :meth:`~taurex.data.profiles.chemistry.autochemistry.AutoChemistry.gases`




.. _decorator: https://docs.python.org/3/library/functions.html#property
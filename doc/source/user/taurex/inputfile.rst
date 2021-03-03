.. _inputfile: 

=================
Input File Format
=================


Headers
-------

The input file format is fairly simple to work with. The extension
for Taurex3 input files is ``.par`` however this is generally not enforced by the code.
The input is defined in various *headers*, with each header having variables that can be set::

    [Header]
    parameter1 = value
    parameter2 = anothervalue

    [Header2]
    parameter1 = adifferentvalue
Of course comments are handled with ``#``

 The available headers are:
    - :ref:`userglobal`
    - :ref:`userchemistry`
    - :ref:`usertemperature`
    - :ref:`userpressure`
    - :ref:`userplanet`
    - :ref:`userstar`
    - :ref:`usermodel`
    - :ref:`userobservation`
    - :ref:`userbinning`
    - :ref:`userinstrument`
    - :ref:`useroptimizer`
    - :ref:`userfitting`

Not all of these headers are required in an input file. Some will generate
default profiles when not present. To perform retrievals, 
:ref:`userobservation`, :ref:`useroptimizer` and :ref:`userfitting` *MUST*
be present


Some of these may define additional *subheaders* given by the ``[[Name]]`` notation::

    [Header]
    parameter1 = value
        [[Subheader]]
        parameter2 = anothervalue

Variables
---------

String variables take this form::

    #This is valid
    string_variable = Astringvariable 
    #This is also valid
    string_variable_II = "A string variable"

Floats and ints are simply::

    my_int_value = 10

And lists/arrays are defined using commas::

    my_int_list = 10,20,30
    my_float_list = 1.1,1.4,1.6,
    my_string_list = hello,how,are,you


Dynamic variables
-----------------

The input file is actually a dynamic format and its available variables can change depending
on the choice of certain profiles and types. For example lets take the ``[Temperature]`` header,
it contains the variable ``profile_type`` which describes which temperature profile to use. 
Setting this to ``isothermal`` gives us the ``T`` variable which defines the isothermal temeprature::

    [Temperature]
    profile_type = isothermal
    T = 1500.0

Now if we change the profile type to ``guillot2010`` it will use the Guillot 2010 temperature profile
which gives access to the variables ``T_irr``, ``kappa_irr``, ``kappa_v1``, ``kappa_v2``  and ``alpha``
instead::

    [Temperature]
    profile_type = guillot2010
    T_irr=1500
    kappa_irr=0.05
    kappa_v1=0.05
    kappa_v2=0.05
    alpha=0.005

However setting ``T`` will throw an error as it doesn't exist anymore::

    [Temperature]
    profile_type = guillot2010
    #Error is thrown here
    T=1500
    kappa_irr=0.05

This also applies to fitting parameters, profiles provide certain fitting parameters
and changing the model means that these parameters may not exist anymore.


Mixins
------

.. versionadded:: 3.1

:ref:`mixin` can be applied to any base component through the ``+``
operator::

    [Temperature]
    profile_type = mixin1+mixin2+base

Where we apply ``mixin1`` and ``mixin2`` to a ``base``.
Including mixins will also include their keywords as well. If ``mixin1``
has the keyword ``param1``, ``mixin2`` has ``param2`` and ``base`` has
``another_param`` then we can define in the input file::

    [Temperature]
    profile_type = mixin1+mixin2+base
    param1 = "Hello"       # From mixin 1
    param2 = "World"       # From mixin 2 
    another_param = 10.0   # From base

Mixins are evaluated in reverse, the last must be a *non-mixin*
for example if we have a ``doubler`` mixin that doubles temperature profiles 
then this is valid::

    [Temperature]
    profile_type = doubler+isothermal

but this is *not valid*::

    [Temperature]
    profile_type = isothermal+doubler

Additionally we cannot have more than one base class so this is *invalid*::

    [Temeprature]
    profile_type = doubler+isothermal+guillot

The reverse evaluation means that the first mixin will be *applied last*.
If we have another mixin called ``add50`` which adds 50 K to the profile,
then::

    [Temperature]
    profile_type = doubler+add50+isothermal
    T = 1000

Will result in a temperature profile of :math:`2100~K`. If we instead do this::

    [Temperature]
    profile_type = add50+doubler+isothermal
    T = 1000

Then the resultant temperature will be :math:`2050~K`.

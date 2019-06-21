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
    - ``[Global]``
    - ``[Chemistry]``
    - ``[Temperature]``
    - ``[Pressure]``
    - ``[Model]``
    - ``[LightCurve]``
    - ``[Optimizer]``
    - ``[Fitting]``

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
Setting this to ``isothermal`` gives us the ``iso_temp`` variable which defines the isothermal temeprature::

    [Temperature]
    profile_type = isothermal
    iso_temp = 1500.0

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

However setting ``iso_temp`` will throw an error as it doesn't exist anymore::

    [Temperature]
    profile_type = guillot2010
    #Error is thrown here
    iso_temp=1500
    kappa_irr=0.05

This also applies to fitting parameters, profiles provide ceertain fitting parameters
and changing the model means that these parameters may not exist anymore.

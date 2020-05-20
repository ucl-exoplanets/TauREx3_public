.. _userfitting:

=============
``[Fitting]``
=============

This header deals with controlling the fitting procedure.

The format for altering and controlling fitting parameters is of the form::

    fit_param:option = value

Here ``fit_param`` is the name of the fitting parameter as is given
under the *Fitting Parameters* headers in the user documentation. This also
includes any custom fitting parameters provided by a users custom class (See: :ref:`customtypes`)
Only parameters that exist within the forward model can be set/altered. Trying to set
any other parameter will yield an error.


``option`` defines a set of control key words that alter what the fitting parameter does.
For example, we can enable fitting of the planet radius using the ``fit`` option::

    [Fitting]
    planet_radius:fit = True


New-style priors
================

New in TauREx 3.1 is the ``prior`` option which allows you define a
prior for a particular fitting parameter. This replaces the older
method by allowing for more control over what type of function to use.
They are expandable with new ones implemented through plugins or custom code.

Its syntax is very no similar to creating an object in python, for
example to define a uniform prior of bounds 0.8--5.0 Jupiter masses 
we can do::

    [Fitting]
    planet_radius:fit = True
    planet_radius:prior = "Uniform(bounds=(0.8, 5.0))"

It is **important** that the prior definition is surrounded by quotation
marks. The prior definitions can contain multiple and distinct arguments,
and have seperate ``Log`` forms as well with arguments in log-space::

    [Fitting]
    H2O:fit = True
    H2O:prior = "LogUniform(bounds=(-12, -2))"

Often these log-forms have extra linear (``lin``) arguments where
they are defined in linear space instead, for example, the
prior space::

    [Fitting]
    H2O:fit = True
    H2O:prior = "LogUniform(lin_bounds=(1e-12, 1e-2))"

is equivalent to the previous example. 
The second included prior is the ``Gaussian`` prior which 
has mean and standard deviation arguments::

    planet_radius:prior = "Gaussian(mean=1.0,std=0.3)"

as well as log versions::

    H2O:prior = "LogGaussian(mean=-4,std=2)"

The mean can be defined in linear space with the ``lin_mean``
argument::

    H2O:prior = "LogGaussian(lin_mean=1e-4,std=2)"



Old-Style priors
================

.. warning::
    It is recommended that the new style priors are used.
    These are only included for compatability and will be removed in
    the next major version of TauREx

We can set the prior boundaries between 1.0 - 5.0 Jupiter masses 
using the ``bounds`` option::

    [Fitting]
    planet_radius:fit = True
    planet_radius:bounds = 1.0, 5.0

And fit it in log space using the ``mode`` option::

    [Fitting]
    planet_radius:fit = True
    planet_radius:bounds = 1.0, 5.0
    planet_radius:mode = log

.. caution::

    ``bounds`` *must* be given in linear space. Even if fitting
    in log space. TauREx3 will automatically convert these bounds to
    the correct fitting space.

If we have a constant H2O chemistry in the atmosphere we can
fit it in linear space instead of the default log::

    [Fitting]
    planet_radius:fit = True
    planet_radius:bounds = 1.0, 5.0
    planet_radius:mode = log
    H2O:fit = True
    H2O:mode = linear
    H2O:bounds = 1e-12, 1e-1



Deperecated Options table
==========================

A summary all valid ``option`` is given here:

+------------+-----------------------------------+-----------------------+
| Option     | Description                       | Values                |
+------------+-----------------------------------+-----------------------+
| ``fit``    | Enable or disable fitting         | ``True`` or ``False`` |
+------------+-----------------------------------+-----------------------+
| ``bounds`` | Prior boundaries in linear space  | *min*, *max*          |
+------------+-----------------------------------+-----------------------+
| ``factor`` | Scaled boundaries in linear space | *sclmin*, *sclmax*    |
+------------+-----------------------------------+-----------------------+
| ``mode``   | Fitting space                     | ``log`` or ``linear`` |
+------------+-----------------------------------+-----------------------+


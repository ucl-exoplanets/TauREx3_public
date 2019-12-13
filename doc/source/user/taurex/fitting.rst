
=============
``[Fitting]``
=============

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



Options table
=============

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


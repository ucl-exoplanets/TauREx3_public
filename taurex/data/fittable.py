
"""
This module relates to defining fitting parameters in TauREx3
"""

from functools import partial


def fitparam(f=None, param_name=None, param_latex=None,
             default_mode='linear', default_fit=False, 
             default_bounds=[0.0, 1.0]):
    """
    A decorator used in conjunction with :class:`Fittable` to inform which
    parameters can be fit and its properties. On its own it acts like the
    ``property`` decorator. When used within a :class:`Fittable` class it
    serves to tag a property as able to fit and allows the
    class to compile all parameters that can be fit.

    Its usage is simple, simply wrap a method and define its properties:

    .. code-block:: python

        class Foo(Fittable):

            @fitparam(param_name='foobar',param_latex='$Foo^{bar}$')
            def bar(self):
                return 'Foobar'

            @bar.setter
            def bar(self,value):
                self.value = value


    Parameters
    ----------
    f: function
        Function being passed. Automatically done when used as a decorator

    param_name: str
        Nicer name of the parameter. Referenced by the optimizer.

    param_latex: str
        Latex version of the parameter name, useful in plotting and
        making figures

    default_mode: ``linear`` or ``log``
        Defines how the optimizer should read and write the parameter.
        ``linear`` reads/write everything as is.
        ``log`` informs the optimizer to transform from native->log space
        when read and to transfrom log->native when
        writing. This also applies to the boundaries

    default_fit: bool
        Whether this is included in the fit without the user explicity saying
        so (Default: False)

    default_bounds: :obj:`list`
        Default minimum and maximum fitting boundary. Must always be defined
        in linear space


    """
    if f is None:
        return partial(fitparam, param_name=param_name,
                       param_latex=param_latex,
                       default_mode=default_mode,
                       default_fit=default_fit,
                       default_bounds=default_bounds)

    def wrapper(self, *args, **kwargs):

        return f(self, *args, **kwargs)
    wrapper.param_name = param_name
    wrapper.param_latex = param_latex
    wrapper.default_fit = default_fit
    wrapper.default_bounds = default_bounds
    wrapper.default_mode = default_mode
    wrapper.decorated = 'fitparam'
    pwrap = property(wrapper)

    return pwrap


class Fittable(object):
    """

    A class that manages fitting parameters. Not really used on its own
    it should really be inherited from to be used properly. It also provides
    class with the ability to read and write fitting parameters using their 
    params names, for example, if we create a class like this:

    .. code-block:: python

        class Foo(Fittable):

            def __init__(self):
                self.value = 10

            @fitparam(param_name='foobar',param_latex='$Foo^{bar}$',default_bounds=[1,12])
            def bar(self):
                return self.value

            @bar.setter
            def bar(self,value):
                self.value = value

    We can read and write data in the standard python way like so:

    >>> foo = Foo()
    >>> foo.bar
    10
    >>> foo.bar = 20
    >>> foo.bar
    20

    but we also get this functionality for free:

    >>> foo['foobar']
    20
    >>> foo['foobar'] = 30
    >>> foo['foobar']
    30


    """

    def __init__(self):
        self._param_dict = {}

        self.compile_fitparams()

    def add_fittable_param(self, param_name, param_latex, fget, fset,
                           default_mode, default_fit, default_bounds):
        """
        Adds a fittable parameter to the internal dictionary. Used during init
        to add all :func:`fitparam` decorated methods and can also be utilized
        by a user to manually add new fitting parameters. This is
        useful for giving fitting parameters names that depend on certain
        attributes (e.g. molecule name in a gas profile
        see
        :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`)
        or when converting lists into fitting parameters
        (e.g. Normalization factor in light curves
        see:
        :class:`~taurex.model.lightcurve.lightcurve.LightCurveModel` )

        Parameters
        ----------
        param_name: str
            Nicer name of the parameter. Referenced by the optimizer.

        param_latex: str
            Latex version of the parameter name, useful in plotting and making
            figures

        fget: function
            a function that returns the value of the parameter

        fset: function
            a function the writes the value of the parameter

        default_mode: ``linear`` or ``log``
            Defines how the optimizer should read and write the parameter.
            ``linear`` reads/write everything as is.
            ``log`` informs the optimizer to transform from native->log space
            when read and to transfrom log->native when writing. This also
            applies to the boundaries

        default_fit: bool
            Whether this is included in the fit without the user explicity
            saying so (Default: False)

        default_bounds: :obj:`list`
            Default minimum and maximum fitting boundary. Must always be
            defined in the native space

        """
        if param_name in self._param_dict:
            raise AttributeError(
                'param name {} already exists'.format(param_name))

        self._param_dict[param_name] = (param_name,
                                        param_latex,
                                        fget.__get__(self),
                                        fset.__get__(self),
                                        default_mode,
                                        default_fit,
                                        default_bounds)

    def compile_fitparams(self):
        """
        Loops through and finds all fitting parameters in the class and adds
        it to the internal dictionary
        """
        for fitparams in self.find_fitparams():

            get_func = fitparams.fget
            set_func = fitparams.fset
            param_name = get_func.param_name
            param_latex = get_func.param_latex
            def_mode = get_func.default_mode
            def_fit = get_func.default_fit
            def_bounds = get_func.default_bounds
            self.add_fittable_param(param_name, param_latex, get_func,
                                    set_func, def_mode, def_fit, def_bounds)

    def modify_bounds(self, parameter, new_bounds):
        """
        Modifies the fitting boundary of a parameter

        Parameters
        ----------
        parameter : str
            Name of parameter (given by ``param_name`` in :func:`fitparam`)

        new_bounds : :obj:`list`
            New minimum and maximum fitting boundaries.

        """
        name, latex, fget, fset, mode, to_fit, bounds = \
            self._param_dict[parameter]

        bounds = new_bounds

        self._param_dict[parameter] = \
            name, latex, fget, fset, mode, to_fit, bounds

    def __getitem__(self, key):
        param = self._param_dict[key]

        return param[2]()

    def __setitem__(self, key, value):
        return self._param_dict[key][3](value)

    def find_fitparams(self):
        """
        Finds and returns fitting parameters

        Yields
        ------
        method : function
            class method that is defined with the :func:`fitparam` decorator
        """

        for klass in self.__class__.mro():

            for method in klass.__dict__.values():
                if hasattr(method, 'fget'):
                    prop = method.fget
                    if hasattr(prop, 'decorated'):
                        if prop.decorated == 'fitparam':
                            yield method

    def fitting_parameters(self):
        """
        Returns all fitting parameters found as a dictionary

        Returns
        -------
        params : :obj:`dict`
            Dictionary with key as the parameter name (``param_name``) 
            and value as a tuple with:
                * parameter name
                * parameter name in Latex form
                * get function
                * set function
                * fitting scale
                * fit as default
                * fitting boundaries

        """
        return self._param_dict

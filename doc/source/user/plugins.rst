=======
Plugins
=======


Plugins are a new feature since version 3.1. Inspired by Flask extensions, plugins are extra packages that add
new functionality to TauREx. They allow anyone to improve and expand TauRExs capabilities
without modifying the main codebase. For example, new forward models, opacity formats,
chemistry and optimizers.

Finding Plugins
===============

TauREx plugins usually are named as 'taurex_foo' or 'taurex-bar'. The Plugin Catalogue contains 
a list of plugins developed by us here :ref:`pluginscata` . You can also
search PyPI for packages tagged with `Framework :: TauREx <pypi_>`_.


Using Plugins
=============

Consult each plugins documentation for installation and usage. Generally TauREx
searches for entry points in ``taurex.plugins`` and adds each component into the 
correct point in the pipeline

Lets take chemistry for example. Assuming a fresh install, 
we can see what is available to use in TauREx 3 by writing in the command prompt::

    taurex --keywords chemistry

We get the output::

    ╒══════════════════╤═════════════════╤══════════╕
    │ chemistry_type   │ Class           │ Source   │
    ╞══════════════════╪═════════════════╪══════════╡
    │ file / fromfile  │ ChemistryFile   │ taurex   │
    ├──────────────────┼─────────────────┼──────────┤
    │ taurex / free    │ TaurexChemistry │ taurex   │
    ╘══════════════════╧═════════════════╧══════════╛

We only have chemistry from a ``file`` and ``free`` chemistry. Supposing we wish to make use of FastChem.
In the previous version we could easily load in an output from FastChem but what
if we wanted to perform retrievals on the chemistry? We would need to write a wrapper of somekind
that loads the C++ library into python before blah blah blah. A considerable amount of effort
and likely someone else has solved the problem beforehand.
This is what plugins solve!

With 3.1, we can now install the full FastChem chemistry code into TauREx3 with a single command::

    pip install taurex_fastchem

Easy!

Now if we check the available chemistries we see::

    ╒══════════════════╤═════════════════╤══════════╕
    │ chemistry_type   │ Class           │ Source   │
    ╞══════════════════╪═════════════════╪══════════╡
    │ file / fromfile  │ ChemistryFile   │ taurex   │
    ├──────────────────┼─────────────────┼──────────┤
    │ fastchem         │ FastChem        │ fastchem │
    ├──────────────────┼─────────────────┼──────────┤
    │ taurex / free    │ TaurexChemistry │ taurex   │
    ╘══════════════════╧═════════════════╧══════════╛

We now have FastChem available!!! 

.. tip::
    It must be stressed that *downloading and installing FastChem is not necessary*, 
    the plugin includes the precompiled library in the package.

Now we can use FastChem in the input file with retrievals::

    [Chemistry]
    chemistry_type = fastchem
    selected_elements = H, He, C, N, O
    metallicity = 2

    [Fitting]
    C_O_ratio:fit = True
    C_O_ratio:priors = "LogUniform(-1,2)"


Building Plugins
================

While `PyPI <pypi_>`_ contains a growing list of TauREx plugins,
you may not find a plugin that matches your needs. In this case
you can try building your own! Read :ref:`buildplugin` to learn how
to develop your own and extend TauREx!


.. _pypi: https://pypi.org/search/?c=Framework+%3A%3A+TauREx
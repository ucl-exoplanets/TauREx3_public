.. _quickstart:

==========
Quickstart
==========


To get quickly up to speed lets try an example run using TauREx3. We will be using the ``examples/quickstart.par``
file as a starting point and ``examples/test_data.dat`` as our observation

Prerequisites
-------------

Before reading this you should have a few things on hand. Firstly ``H2O`` and ``CH4`` absorption cross sections in a python
pickle format are required. Secondly some collisionally induced absorption (CIA) cross sections are also
required for a later part for ``H2-He`` and ``H2-H2``, you can get these from the HITRAN_ site. 

Setup
------
In order to begin running forward models we need to tell *TauREx3* where our cross-sections are.
We can do this by defining an ``xsec_path`` for cross sections and ``cia_path`` for CIA cross-sections under the
``[Global]`` header in our ``quickstart.par`` files like so::

    [Global]
    xsec_path = /path/to/xsec
    cia_path = /path/to/cia


Forward Model
-------------

Using our input we can run and plot the forward model by doing::

    taurex -i quickstart.par --plot

And we should get:

.. figure::  _static/firstfm.png
   :align:   center

   Our first forward model

Lets try plotting it against our observation. Under the ``[Observation]`` header
we can add in the ``observed_spectrum`` keyword and point it to our ``test_data.dat`` file like so::

    [Observation]
    observed_spectrum = /path/to/test_data.dat

Now the spectrum will be binned to our observation:

.. figure:: _static/fm_obs_bin.png

           ``grid_type = observed``


You may notice that general structure and peaks don't seem to match up with observation.
Our model doesn't seem to do the job and it may be the fault of our choice of molecule. Lets move on to chemistry.


Chemistry
---------

As we've seen, ``CH4`` doesn't fit the observation very well, we should try adding in another molecule.
Underneath the ``[Chemistry]`` section we can add another sub-header with the name of our molecule, for this 
example we will use a ``constant`` gas profile which keeps it abundance constant throughout the atmosphere,
there are other more complex profiles but for now we'll keep it simple::

    [Chemistry]
    chemistry_type = taurex
    fill_gases = H2,He
    ratio=4.8962e-2

        [[H2O]]
        gas_type = constant
        mix_ratio=1.1185e-4

        [[CH4]]
        gas_type=constant
        mix_ratio=1.1185e-4

        [[N2]]
        gas_type = constant
        mix_ratio = 3.00739e-9

Plotting it gives:

.. image::  _static/ch4_and_h2o.png

We're getting there. It looks like H2O is definately there but maybe ``CH4`` isn't? Lets try it
by commenting it out::

    [Chemistry]
    chemistry_type = taurex
    fill_gases = H2,He
    ratio=4.8962e-2

        [[H2O]]
        gas_type = constant
        mix_ratio=1.1185e-4

        #[[CH4]]
        #gas_type=constant
        #mix_ratio=1.1185e-4

        [[N2]]
        gas_type = constant
        mix_ratio = 3.00739e-9

.. image::  _static/h2o_only.png

Much much better! We're still missing something though...

Contributions
-------------

It seems moelcular absorption is not the only process happening in the atmosphere. Looking at the shorter
wavelengths we see the characteristic behaviour of **Rayleigh scattering** and a little from **collisionally**
**induced** **absorption**. We can easily add these contributions under the ``[Model]`` section of the input file.
Each *contribution* is represented as a subheader with additional arguments if necessary. By default we have
contributions from molecular ``[[Absorption]]`` 
Lets add in some ``[[CIA]]`` from ``H2-H2`` and ``H2-He`` and ``[[Rayleigh]]`` scattering to the model::

    [Model]
    model_type = transmission

        [[Absorption]]

        [[CIA]]
        cia_pairs = H2-He,H2-H2

        [[Rayleigh]]

.. image::  _static/ray_and_cia.png

Hey not bad!! It might be worth seeing how each of these processes effect the spectrum. Easy, we can run
``taurex`` with the ``-c`` argument which plots the basic contributions::

    taurex -i quickstart.par --plot -c

.. image::  _static/contrib.png


If you want a more detailed look of the each contribution you can use the ``-C`` option instead::

    taurex -i quickstart.par --plot -C

.. image::  _static/full_contrib.png

Pretty cool. We're almost there. Lets save what we have now to file.

Storage
-------

``Taurex3`` uses the HDF5_ format to store its state and results. We can accomplish this by 
using the ``-o`` output argument::

    taurex -i quickstart.par --plot -c -o myfile.hdf5

``HDF5`` has many viewers such as HDFView_ or HDFCompass_ and APIs such as Cpp_, FORTRAN_ and Python_.
Pick your poison.


Retrieval
---------

So we're close to the observation but not quite there and I suspect its the 
temperature profile. We should try running a retrieval. We will use nestle_ as our optimizer of choice
but other brands are available. This has already be setup under the ``[Optimizer]`` section of the input 
file so we will not worry about it now. We now need to inform the optimizer what parameters we need to fit.
The ``[Fitting]`` section should list all of the parameters in our model that we want (or dont want) to fit 
and *how* to go about fitting it. By default the ``planet_radius`` parameter is fit when no section is provided,
we should start by creating our ``[Fitting]`` section and disabling the ``planet_radius`` fit::
    
    [Fitting]
    planet_radius:fit = False

the syntax is pretty simple, its essentially ``parameter_name:option`` with ``option`` being either 
``fit``, ``bounds`` and ``mode``. ``fit`` is simply tells the optimizer whether to fit the parameter, ``bounds``
describes the parameter space to optimize in and ``mode`` instructs the optimizer to fit in either ``linear``
or ``log`` space.
The parameter we are interested in is isothermal temperature which is represented as ``T``, and we will fit
it within *1200 K* and *1400 K*::

    [Fitting]
    planet_radius:fit = False
    T:fit = True
    T:bounds = 1200.0,1400.0

We don't need to include ``mode`` as by default ``T`` fits in linear space. Some parameters such as
abundances fit in log space by default.

Running taurex like before will just plot our forward model. To run the retrieval we simply add
the ``--retrieval`` keyword like so::

    taurex -i quickstart.par --plot -o myfile.hdf5 --retrieval

We should now see something like this pop up::

    -------------------------------------
    ------Retrieval Parameters-----------
    -------------------------------------

    Dimensionality of fit: 1

    Param      Value    Bound-min    Bound-max
    -------  -------  -----------  -----------
    T        1265.98         1200         1400

    taurex.Nestle - WARNING - Beginning fit......
    WARNING:taurex.Nestle:Beginning fit......
    it=   125 logz=1838.310559

It should only take a few minutes to run. Once done we should get an output like this::

    ------------------------------
    -------Retrieval output-------
    ------------------------------

    Parameter      Value    Sigma
    -----------  -------  -------
    T            1360.31  3.55803

So the temperature should have been *1360 K*, huh, and lets see how it looks:

.. image::  _static/retrieval.png

.. image:: _static/delicious.jpg

Oh and its saved to our HDF5 file under the ``Fit`` header with all the weights, traces and results.



.. _HITRAN: https://hitran.org/cia/

.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/

.. _HDFView: https://www.hdfgroup.org/downloads/hdfview/

.. _nestle: https://github.com/kbarbary/nestle

.. _HDFCompass: https://support.hdfgroup.org/projects/compass/

.. _FORTRAN: https://support.hdfgroup.org/HDF5/doc/fortran/index.html

.. _Cpp: https://support.hdfgroup.org/HDF5/doc/cpplus_RM/index.html

.. _Python: https://www.h5py.org/
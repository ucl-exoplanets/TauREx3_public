===
MPI
===

**The message passing protocol (MPI) is not needed to install,
run and perform retrievals.**

**This is more of a help guide to getting mpi4py working 
successfully and is not specific to TauREx3**

There are some optimizers
that can make use of MPI to significantly speed up retrievals.
Specifically the Multinest and PolyChord optimizers. Considering
most people have difficulty with installing it, this guide
has been written to make the experience as smooth as possible.

First you must have an MPI library installed, this may already
be installed in your system (such as a cluster) 
or you can install a library youself.
For Mac users the quickest way to install it is through Homebrew::

    brew install openmpi

Now we need to install our python MPI wrapper library mpi4py_::

    pip install mpi4py


You can test the installation::

    mpirun -n 4 python -m mpi4py.bench helloworld

Replace *mpirun* with whatever the equivalent is for your
system

You should get a similar output like so::

    Hello, World! I am process 0 of 4 on blahblah.
    Hello, World! I am process 1 of 4 on blahblah.
    Hello, World! I am process 2 of 4 on blahblah.
    Hello, World! I am process 3 of 4 on blahblah.

Then you are all set! Theres no need to reinstall TauREx3
as it will now import it successfully when run. 

.. tip::

    TauREx3 actually
    suppresses text output from other processes so running under MPI
    will actually look its being run serially. In fact if you
    get multiple of the same outputs this is a surefire way to
    know that something is wrong with the mpi4py installation!!!

**However** if you get something like this::

    Hello, World! I am process 0 of 1 on blahblah.
    Hello, World! I am process 0 of 1 on blahblah.
    Hello, World! I am process 0 of 1 on blahblah.
    Hello, World! I am process 0 of 1 on blahblah.

This means mpi4py has not correctly installed. This likely happens
in cluster environments with multiple MPI libraries. You can overcome
this by re-installing mpi4py with the ``MPICC`` enviroment set::

    env MPICC=mpicc pip install --no-cache-dir mpi4py

Or::

    env MPICC=/path/to/mpicc pip install --no-cache-dir mpi4py

Now re-run the test. If you get the correct result. Horray! If not,
its best to ask your administrator.

Once you have this installed, you can install pymultinest_ here.


.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/index.html
.. _pymultinest: https://johannesbuchner.github.io/PyMultiNest/install.html
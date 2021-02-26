.. _basics:

======
Basics
======

There are some common rules when developing new components for TauREx 3.
These apply for the majority of components in the TauREx pipeline. The 
only major exception is the :class:`~taurex.opacity.opacity.Opacity` related
classes which have a different system in place.


Automatic Input Arguments
-------------------------

In TauREx, when loading in a class, it will dynamically
parse all ``__init__`` arguments and make them accessible in the input file.
If you build a new temperature class:

.. code-block:: python

    class MyNewTemperatureProfile(TemperatureProfile):

        def __init__(self, mykeyword=[1,2,3], another_keyword='A string'):
            print('A: ',mykeyword)
            print('B: ',another_keyword)

        @classmethod
        def input_keywords(cls):
            return ['myprofile']

Then the keyword arguments ``mykeyword`` and ``another_keyword`` become arguments
in the input file::

    [Temperature]
    profile_type = myprofile
    my_keyword = 5,6,7,
    another_keyword = "Another string"

Which when run will produce::

    A: [5, 6, 7]
    B: Another string

We recommend defining all ``__init__`` arguments as keywords if you intend
for your components to be used through the input file.


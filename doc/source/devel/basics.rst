.. _basics:

======
Basics
======

There are some common rules when developing new components for TauREx 3.
These apply for the majority of components in the TauREx pipeline. The 
only major exception is the :class:`~taurex.opacity.opacity.Opacity` related
classes which have a different system in place.


Automatic Input Arguments
=========================

In TauREx, when loading in a class, it will dynamically
parse all ``__init__`` arguments and make them accessible in the input file.
If you build a new temperature class:

.. code-block:: python

    from taurex.temperature import TemperatureProfile
    import numpy as np

    class MyNewTemperatureProfile(TemperatureProfile):

        def __init__(self, mykeyword=[1,2,3], another_keyword='A string'):
            super().__init__(name=self.__class__.__name__)
            print('A: ',mykeyword)
            print('B: ',another_keyword)

        @property
        def profile(self):
            T = np.random.rand(self.nlayers)*1000 + 1
            return T

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


Bibliography
============

.. versionadded:: 3.1

It is important to recognise works involved for each component during a TauREx run.
TauREx includes a basic bibliography system that will collect and parse bibtex entries
embedded in each component.

Embedding bibliographic information for most cases only requires defining the ``BIBTEX_ENTRIES``
class variable as a list of bibtex entries:

.. code-block:: python

    from taurex.temperature import TemperatureProfile
    import numpy as np

    class MyNewTemperatureProfile(TemperatureProfile):

        def __init__(self, mykeyword=[1,2,3], another_keyword='A string'):
            super().__init__(name=self.__class__.__name__)
            print('A: ',mykeyword)
            print('B: ',another_keyword)

        @property
        def profile(self):
            T = np.random.rand(self.nlayers)*1000 + 1
            return T

        @classmethod
        def input_keywords(cls):
            return ['myprofile']
        
        BIBTEX_ENTRIES = [
            """
            @article{myprof,
                url = {https://vixra.org/abs/1512.0013},
                year = 2015,
                month = {dec},
                volume = {1512},
                number = {0013},
                author = {Ben S. Dover, Micheal T Hunt, Christopher S Peacock},
                title = {A New Addition to the Stellar Metamorphsis. the Merlin Hypothesis},
                journal = {vixra},
            }
            """,
            """
            @misc{vale2014bayesian,
                title={Bayesian Prediction for The Winds of Winter}, 
                author={Richard Vale},
                year={2014},
                eprint={1409.5830},
                archivePrefix={arXiv},
                primaryClass={stat.AP}
            }
            """

        ]

.. warning::

    If your bibtex entry includes non-unicode characters then Python will refuse
    to run or your plugin may not be able to load into the TauREx pipeline.

Running TauREx, on program end we get::

    A New Addition to the Stellar Metamorphsis. the Merlin Hypothesis
    Ben S. Dover, Micheal T Hunt, Christopher S Peacock
    vixra, 1512, dec, 2015

    Bayesian Prediction for The Winds of Winter
    Vale, Richard
    arXiv, 1409.5830, 2014

Additionally, running ``taurex`` with ``--bibtex mybib.bib`` will
export the citation as a ``.bib`` file::

    @misc{cad6f055,
        author = "Al-Refaie, Ahmed F. and Changeat, Quentin and Waldmann, Ingo P. and Tinetti, Giovanna",
        title = "TauREx III: A fast, dynamic and extendable framework for retrievals",
        year = "2019",
        eprint = "1912.07759",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.IM"
    }

    @article{6720c2d1,
        author = "Ben S. Dover, Micheal T Hunt, Christopher S Peacock",
        url = "https://vixra.org/abs/1512.0013",
        year = "2015",
        month = "dec",
        volume = "1512",
        number = "0013",
        title = "A New Addition to the Stellar Metamorphsis. the Merlin Hypothesis",
        journal = "vixra"
    }

    @misc{f55ed081,
        author = "Vale, Richard",
        title = "Bayesian Prediction for The Winds of Winter",
        year = "2014",
        eprint = "1409.5830",
        archivePrefix = "arXiv",
        primaryClass = "stat.AP"
    }

Bibliographies are additive as well, if we decided to build on top of this class
we do not need to redefine the older bibliographic information as all parent
bibliographic information is also inherited:

.. code-block:: python:

    class AnotherProfile(MyNewTemperatureProfile):
        # ...

        BIBTEX_ENTRIES = [
            """
            @misc{scott2015farewell,
                title={A Farewell to Falsifiability}, 
                author={Douglas Scott and Ali Frolop and Ali Narimani and Andrei Frolov},
                year={2015},
                eprint={1504.00108},
                archivePrefix={arXiv},
                primaryClass={astro-ph.CO}
            }
        ]

Will yield::

    A Farewell to Falsifiability
    Douglas Scott, Ali Frolop, Ali Narimani, Andrei Frolov
    arXiv, 1504.00108, 2015

    A New Addition to the Stellar Metamorphsis. the Merlin Hypothesis
    Ben S. Dover, Micheal T Hunt, Christopher S Peacock
    vixra, 1512, dec, 2015

    Bayesian Prediction for The Winds of Winter
    Vale, Richard
    arXiv, 1409.5830, 2014


You can get citations from each object through the :py:meth:`~taurex.data.citation.Citable.citations`
method which will output a :obj:`list` of parsed bibtex entries::

    >>> t = MyNewTemperatureProfile()
    >>> t.citations()
    [Entry('article',
   fields=[
     ('url', 'https://vixra.org/abs/1512.0013'), 
     ('year', '2015'), 
     ('month', 'dec'), 
     ('volume', '1512'), 
     ('number', '0013'), 
     ('title', 'A New Addi.....etc



A printable string can also be generated
using the :meth:`~taurex.data.citation.Citable.nice_citation`
method::

    >>> print(t.nice_citation())
    A New Addition to the Stellar Metamorphsis. the Merlin Hypothesis
    Ben S. Dover, Micheal T Hunt, Christopher S Peacock
    vixra, 1512, dec, 2015

    Bayesian Prediction for The Winds of Winter
    Vale, Richard
    arXiv, 1409.5830, 2014


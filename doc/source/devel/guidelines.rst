.. _coding-guide:

===============================
Taurex 3 development guidelines
===============================

Overview
---------

Here we describe the development guidelines for 
TauREx 3 and some advice for those wishing to contribute.
Since TauREx 3 is open-source, all contributions are welcome!!!

Development on TauREx 3 should be focused on building and improving the framework.
New components (i.e. chemistries, profiles etc.) are generally not built directly
into the TauREx 3 codebase.

We recommend building new components as Plugins. You can refer to the :ref:`buildplugin`
guide.


Documentation
-------------

All standalone documentation should be written 
in plain text (``.rst``) files using reStructuredText_ 
for markup and formatting.
All docstrings should follow the numpydoc_ format.
New features must include both appropriate docstrings
and any necessary standalone documentation

Unit-testing
------------

Unittesting is important in preserving sanity
and code integrity. For TauREx 3 we employ
pytest_. When bugfixing, ensure 
unittests pass. In the root directory do::

    pytest tests/

To run all unit tests in TauREx3

When building new features, create new unittests and
include them in the ``test/`` directory,
any future collaborations from other developers are less
likely to break your feature unexpectedly when they have
something to easily test against.

Some rules:

- No `extra` files should be included. Instead
  have the unit test generate them on the spot.

- We recommended hypothesis_ for bug finding

Coding conventions
==================

Code should follow the PEP8_ standard. This can be
facilitated with a linter such as flake8_

Source control
==============

Git is the source control environment used.
In particular, we follow the git-flow_ branching model internally.
In this model, there are two long-lived branches:

- ``master``: used for official releases. **Contributors should 
  not need to use it or care about it**

- ``develop``: reflects the latest integrated changes for the next 
  release. This is the one that should be used as the base for 
  developing new features or fixing bugs.


For contributions we employ the Fork-and-Pull_ model:

1. A contributor first forks_ the TauREx3 repo
2. They then clone their forked branch
3. The contributor then commits and merges their changes into
   their forked ``develop`` branch
4. A Pull-Request_ is created against the official ``develop``
   branch
5. Anyone interest can review and comment on the pull request,
   and suggest changes. The contributor can continue to commit more
   changes until it is approved
6. Once approved, the code is considered ready and the pull request
   is merged into the official develop


.. _reStructuredText:  http://docutils.sourceforge.net/rst.html
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/
.. _PEP8: http://www.python.org/peps/pep-0008.html
.. _flake8: http://flake8.pycqa.org/en/latest/
.. _git-flow: https://danielkummer.github.io/git-flow-cheatsheet/
.. _Fork-and-Pull: https://en.wikipedia.org/wiki/Fork_and_pull_model
.. _forks: https://help.github.com/articles/fork-a-repo/
.. _Pull-Request: https://help.github.com/articles/creating-a-pull-request/
.. _pytest: https://docs.pytest.org/en/stable/
.. _hypothesis: https://hypothesis.readthedocs.io/en/latest/quickstart.html
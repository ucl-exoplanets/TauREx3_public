---
title: Taurex 3 development guidelines
---

Overview
========

Here we describe the development guidelines for TauREx 3 and some advice
for those wishing to contribute. Since TauREx 3 is open-source, all
contributions are welcome!!!

Documentation
=============

All standalone documentation should be written in plain text (`.rst`)
files using [reStructuredText](http://docutils.sourceforge.net/rst.html)
for markup and formatting. All docstrings should follow the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/) format. New
features must include both appropriate docstrings and any necessary
standalone documentation

Unit-testing
============

Unittesting is important in preserving sanity and code integrity. For
TauREx 3 we employ the standard
[unittest](https://docs.python.org/3/library/unittest.html) module. When
bugfixing, ensure unittests pass. In the root directory do:

    python -m unittest discover

To run all unit tests in TauREx3

When building new features, create new unittests and include them in the
`test/` directory, any future collaborations from other developers are
less likely to break your feature unexpectedly when they have something
to easily test against.

Some rules:

-   Tests must be quick (in the order of seconds)
-   No [extra]{.title-ref} files should be included. Instead have the
    unit test generate them on the spot.

Coding conventions
------------------

Code should follow the [PEP8](http://www.python.org/peps/pep-0008.html)
standard. This can be facilitated with a linter such as
[flake8](http://flake8.pycqa.org/en/latest/)

Source control
--------------

Git is the source control environment used. In particular, we follow the
[git-flow](https://danielkummer.github.io/git-flow-cheatsheet/)
branching model internally. In this model, there are two long-lived
branches:

-   `master`: used for official releases. \*\*Contributors should not
    need to use it or care about it\*\*
-   `develop`: reflects the latest integrated changes for the next
    release. This is the one that should be used as the base for
    developing new features or fixing bugs.

For contributions we employ the
[Fork-and-Pull](https://en.wikipedia.org/wiki/Fork_and_pull_model)
model:

1.  A contributor first
    [forks](https://help.github.com/articles/fork-a-repo/) the TauREx3
    repo
2.  They then clone their forked branch
3.  The contributor then commits and merges their changes into their
    forked `develop` branch
4.  A
    [Pull-Request](https://help.github.com/articles/creating-a-pull-request/)
    is created against the official `develop` branch
5.  Anyone interest can review and comment on the pull request, and
    suggest changes. The contributor can continue to commit more changes
    until it is approved
6.  Once approved, the code is considered ready and the pull request is
    merged into the official develop

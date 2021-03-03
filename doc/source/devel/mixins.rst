.. _mixinsdevel:

======
Mixins
======

.. versionadded:: 3.1

Mixins are lighter components with the sole purpose of giving 
*all* atmospheric components new abilities and features. For the coding
inclined you can see the article `here <wiki_>`_.

Motivation
----------

To understand this, lets take an viable scenario. Imagine you've come up with
an amazing idea. What if all temperature profiles must be doubled to
be physically valid? Incredible! So you begin your Nobel Prize winning
work and begin defining new temperature profiles for each of the
available ones in TauREx3. You create ``isothermal_double``,
``npoint_double``, ``guillot_double`` etc and release it to the amazement of
the public. Someone comes along and develops a super new temperature profile,
lets call it ``supernewtemp``. Well now looks like you'll now have to go back
and implement a ``supernewtemp_double`` but no matter, progress comes with
sacrifice. Now your colleague suggests that adding 50K also improves the profile,
so they implement ``isothermal_50``, ``npoint_50``, ``guillot_50`` and ``supernewtemp_50``. 
Now some people say they want to double it and add 50 so someone must create
``isothermal_double_50``, ``npoint_double_50``, ``guillot_double_50`` and ``supernewtemp_double_50``
and other people want to add 50 and double so now we need to build
``isothermal_50_double``, ``npoint_50_double``, ``guillot_50_double`` and ``supernewtemp_50_double``
and oh no someone just created a brand new temperature profile and deeper into the
endless abyss you go.

This is what mixins solve, if instead we develop a ``doubler`` mixin we can instead
*add* it to our original profile using the ``+`` operator::

    [Temperature]
    profile_type = doubler+isothermal
    T = 1000

And TauREx will build an isothermal profile that doubles itself for you. Neat
We can do the same and build an ``add50`` mixin::

    [Temperature]
    profile_type = add50+isothermal

Now the beauty is that we can stack them together!! If we want to double then add 50
we can write::

    [Temperature]
    profile_type = add50+doubler+isothermal

Or add 50 then double::

    [Temperature]
    profile_type = doubler+add50+isothermal

Mixins also come with their own keywords as well! They can combine themselves with




.. _wiki: https://en.wikipedia.org/wiki/Mixin
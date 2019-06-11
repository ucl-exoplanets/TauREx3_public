"""Just contains a singleton class. Pretty useful"""

class Singleton(object):
    """
    A singleton for your usage. When inheriting do not implement __init__ instead
    override :func:`init`
    
    
    """
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it
    def init(self, *args, **kwds):
        """ Override to act as an init """
        pass

def getattr_recursive(obj, attr):
    split = attr.split('.')
    if len(split) == 1:
        return getattr(obj, split[0])
    else:
        return getattr_recursive(getattr(obj, split[0]), '.'.join(split[1:]))


def setattr_recursive(obj, attr, value):
    split = attr.split('.')
    if len(split) == 1:
        setattr(obj, split[0], value)
    else:
        setattr_recursive(getattr(obj, split[0]), '.'.join(split[1:]), value)


def runfunc_recursive(obj, func, *args, **kwargs):
    split = func.split('.')
    if len(split) == 1:
        return getattr(obj, split[0])(*args, **kwargs)
    else:
        return runfunc_recursive(getattr(obj, split[0]), '.'.join(split[1:]),
                                 *args, **kwargs)

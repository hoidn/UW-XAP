import numpy as np

def npcomp(a, b):
    """
    Compare two iterables, returning true if all elements are
    the same and false otherwise. Nesting is supported.
    """
    fand = lambda x, y: x & y
    if isinstance(a, tuple) or isinstance(a, list):
        return reduce(fand, map(npcomp, a, b))
    elif isinstance(a, str):
        return a == b
    else:
        return np.all(np.isclose(a, b))

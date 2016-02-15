"""
A utils module replicates some of the functionality in utils.py but imports fewer
modules, in order to reduce startup time.
"""

import os
import pkg_resources
from StringIO import StringIO
import numpy as np

PKG_NAME = __name__.split('.')[0]

def resource_f(fpath):
    return StringIO(pkg_resources.resource_string(PKG_NAME, fpath))

def resource_path(fpath):
    return pkg_resources.resource_filename(PKG_NAME, fpath)

def ismpi():
    """
    Check whether this is running in MPI without having to
    initialize mpi4py.
    """
    return 'OMPI' in ' '.join(os.environ.keys())

def ifmpi(func):
    def inner(*args, **kwargs):
        if ismpi():
            return func(*args, **kwargs)
    return inner

def isroot():
    if ismpi():
        from dataccess import utils
        return utils.isroot()
    else:
        return False

def ifroot(func):
    def inner(*args, **kwargs):
        if isroot():
            return func(*args, **kwargs)
    return inner

def is_plottable():
    import config
    if config.noplot:
        return False
    return isroot()

def ifplot(func):
    """
    Decorator that causes a function to execute only if config.noplot is False
    and the MPI core rank is 0.
    """
    def inner(*args, **kwargs):
        import config
        if config.noplot:
            print "PLOTTING DISABLED, EXITING." 
        else:
            @ifroot
            def newfunc(*args, **kwargs):
                return func(*args, **kwargs)
            return newfunc(*args, **kwargs)
    return inner

def flatten_dict(d):
    """
    Given a nested dictionary whose values at the "bottom" are numeric, create
    a 2d array where the rows are of the format:
        k1, k2, k3, value
    This particular row would correspond to the following subset of d:
        {k1: {k2: {k3: v}}}
    Stated another way, this function traverses the dictionary from node to leaf
    once for every single leaf.

    The dict must be "rectangular" (i.e. all leafs are at the same depth)
    """
    def walkdict(d, parents = []):
        if not isinstance(d, dict):
            for p in parents:
                yield p
            yield d
        else:
            for k in d:
                for elt in walkdict(d[k], parents + [k]):
                    yield elt
    def dict_depth(d, depth=0):
        if not isinstance(d, dict) or not d:
            return depth
        return max(dict_depth(v, depth+1) for k, v in d.iteritems())
    depth = dict_depth(d) + 1
    flat_arr = np.fromiter(walkdict(d), float)
    try:
        return np.reshape(flat_arr, (len(flat_arr) / depth, depth))
    except ValueError, e:
        raise ValueError("Dictionary of incorrect format given to flatten_dict: " + e)

@ifroot
def save_0d_event_data(save_path, event_data_dict, **kwargs):
    """
    Save an event data dictionary to file in the following column format:
        run number, event number, value
    """
    dirname = os.path.dirname(save_path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(save_path))
    np.savetxt(save_path, flatten_dict(event_data_dict), **kwargs)

def merge_dicts(*args):
    final = {}
    for d in args:
        final.update(d)
    return final

# Author: O. Hoidn

import numpy as np
import os
import dill
import collections
import pdb
import atexit
from atomicfile import AtomicFile
import pkg_resources
from StringIO import StringIO
from time import time
import hashlib
import inspect

PKG_NAME = __name__.split('.')[0]

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults:
        return dict(zip(args[-len(defaults):], defaults))
    else:
        return {}

def resource_f(fpath):
    return StringIO(pkg_resources.resource_string(PKG_NAME, fpath))

def resource_path(fpath):
    return pkg_resources.resource_filename(PKG_NAME, fpath)

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]
            #return 0.
        elif x > xs[-1]:
            return ys[-1]
            #return 0.
        else:
            return interpolator(x)

    def ufunclike(xs):
        try:
            iter(xs)
        except TypeError:
            xs = np.array([xs])
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike

def make_hashable(obj):
    """
    return hash of an object that supports python's buffer protocol
    """
    return hashlib.sha1(dill.dumps(obj)).hexdigest()

def hashable_dict(d):
    """
    try to make a dict convertible into a frozen set by 
    replacing any values that aren't hashable but support the 
    python buffer protocol by their sha1 hashes
    """
    #TODO: replace type check by check for object's bufferability
    for k, v in d.iteritems():
        # for some reason ndarray.__hash__ is defined but is None! very strange
        #if (not isinstance(v, collections.Hashable)) or (not v.__hash__):
        if isinstance(v, np.ndarray):
            d[k] = make_hashable(v)
    return d

def memoize(timeout = None):
    """
    Memoization decorator with an optional timout parameter.
    """
    cache = {}
    # sad hack to get around python's scoping behavior
    cache2 = {}
    def get_timestamp():
        return cache2[0]
    def set_timestamp():
        cache2[0] = time()

    def decorator(f):
        def new_func(*args):
            if args in cache:
                if (not timeout) or (time() - get_timestamp() < timeout):
                    return cache[args]
            if timeout:
                set_timestamp()
            cache[args] = f(*args)
            return cache[args]
        return new_func
    return decorator

def persist_to_file(file_name):
    """
    Decorator for memoizing function calls to disk

    Inputs:
        file_name: File name prefix for the cache file(s)
    """
    # Optimization: initialize the cache dict but don't load data from disk
    # until the memoized function is called.
    cache = {}

    # These are the hoops we need to jump through because python doesn't allow
    # assigning to variables in enclosing scope:
    state = {'loaded': False, 'cache_changed': False}
    def check_cache_loaded():
        return state['loaded']
    def flag_cache_loaded():
        state['loaded'] = True
    def check_cache_changed():
        return state['cache_changed']
    def flag_cache_changed():
        return state['cache_changed']

    def dump():
        os.system('mkdir -p ' + os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            dill.dump(cache, f)

    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        def compute(key):
            if not check_cache_loaded():
                try:
                    with open(file_name, 'r') as f:
                        to_load = dill.load(f)
                        for k, v in to_load.items():
                            cache[k] = v
                except (IOError, ValueError):
                    print "no cache file found"
                flag_cache_loaded()
            if not key in cache.keys():
                cache[key] = func(*dill.loads(key[0]), **{k: v for k, v in key[1]})
                if not check_cache_changed():
                    # write cache to file at interpreter exit if it has been
                    # altered
                    atexit.register(dump)
                    flag_cache_changed()

        if func.func_code.co_freevars:
            closure_dict = hashable_dict(dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure))))
        else:
            closure_dict = {}

        def new_func(*args, **kwargs):
            # if the "flush" kwarg is passed, recompute regardless of whether
            # the result is cached
            if "flush" in kwargs.keys():
                kwargs.pop("flush", None)
                key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
                compute(key)
            key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
            if key not in cache:
                compute(key)
            return cache[key]
        return new_func

    return decorator

def eager_persist_to_file(file_name, excluded = None):
    """
    Decorator for memoizing function calls to disk.
    Differs from persist_to_file in that the cache file is accessed and updated
    at every call, and that each call is cached in a separate file. This allows
    parallelization without problems of concurrency of the memoization cache,
    provided that the decorated function is expensive enough that the
    additional read/write operations have a negligible impact on performance.

    Inputs:
        file_name: File name prefix for the cache file(s)
    """
    cache = {}

    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        if func.func_code.co_freevars:
            closure_dict = hashable_dict(dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure))))
        else:
            closure_dict = {}

        def gen_key(*args, **kwargs):
            """
            Based on args and kwargs of a function, as well as the 
            closure bindings, generate a cache lookup key
            """
            #return tuple(map(make_hashable, [args, kwargs.items()]))
            # union of default bindings in func and the kwarg bindings in new_func
            merged_dict = get_default_args(func)
            if not merged_dict:
                merged_dict = kwargs
            else:
                for k, v in merged_dict.iteritems():
                    if k in kwargs:
                        merged_dict[k] = kwargs[k]
            if excluded:
                for k in merged_dict.keys():
                    if k in excluded:
                        merged_dict.pop(k)
            key = tuple(map(make_hashable, [args, merged_dict, closure_dict.items()]))
            print "key is", key
#            for k, v in kwargs.iteritems():
#                print k, v
            return key

        def compute(*args, **kwargs):
            local_cache = {}
            file_name = kwargs.pop('file_name', None)
            key = gen_key(*args, **kwargs)
            local_cache[key] = func(*args, **kwargs)
            cache[key] = local_cache[key]
            os.system('mkdir -p ' + os.path.dirname(file_name))
            with open(file_name, 'w') as f:
                dill.dump(local_cache, f)

        def new_func(*args, **kwargs):

            # Because we're splitting into multiple files, we can't retrieve the
            # cache until here
            print "entering ", func.func_name
            full_name = file_name + '_' + str(hash(dill.dumps(args)))
            try:
                with open(full_name, 'r') as f:
                    new_cache = dill.load(f)
                    for k, v in new_cache.items():
                        cache[k] = v
            except (IOError, ValueError):
                print "no cache found"
            # if the "flush" kwarg is passed, recompute regardless of whether
            # the result is cached
            if "flush" in kwargs.keys():
                kwargs.pop("flush", None)
                # TODO: refactor
                compute(*args, file_name = full_name, **kwargs)
            key = gen_key(*args, **kwargs)
            if key not in cache:
                compute(*args, file_name = full_name, **kwargs)
            print "returning from ", func.func_name
            return cache[key]
        return new_func

    return decorator

#def eager_persist_to_file(file_name):
#    """
#    Decorator for memoizing function calls to disk.
#
#    Differs from persist_to_file in that the cache file is accessed and updated
#    at every call, and that each call is cached in a separate file. This allows
#    parallelization without problems of concurrency of the memoization cache,
#    provided that the decorated function is expensive enough that the
#    additional read/write operations have a negligible impact on performance.
#
#    Inputs:
#        file_name: File name prefix for the cache file(s)
#    """
#    cache = {}
#
#    def decorator(func):
#        #check if function is a closure and if so construct a dict of its bindings
#        if func.func_code.co_freevars:
#            closure_dict = hashable_dict(dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure))))
#        else:
#            closure_dict = {}
#        def recompute(key, local_cache, file_name):
#            local_cache[key] = func(*dill.loads(key[0]), **{k: v for k, v in key[1]})
#            os.system('mkdir -p ' + os.path.dirname(file_name))
#            with open(file_name, 'w') as f:
#                dill.dump(local_cache, f)
#
#        def new_func(*args, **kwargs):
#            # Because we're splitting into multiple files, we can't retrieve the
#            # cache until here
#            full_name = file_name + '_' + str(hash(dill.dumps(args)))
#            try:
#                with open(full_name, 'r') as f:
#                    new_cache = dill.load(f)
#                    for k, v in new_cache.items():
#                        cache[k] = v
#            except (IOError, ValueError):
#                print "no cache found"
#            # if the "flush" kwarg is passed, recompute regardless of whether
#            # the result is cached
#            if "flush" in kwargs.keys():
#                kwargs.pop("flush", None)
#                key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
#                # TODO: refactor
#                recompute(key, cache, full_name)
#            key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
#            if key not in cache:
#                recompute(key, cache, full_name)
#            return cache[key]
#        return new_func
#
#    return decorator


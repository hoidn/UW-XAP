import os
import dill
import collections
import pdb
import atexit


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


def persist_to_file(file_name):
    """
    Decorator for memoizing function calls to disk

    Inputs:
        file_name: File name prefix for the cache file(s)
    """
    try:
        with open(file_name, 'r') as f:
            cache = dill.load(f)
    except (IOError, ValueError):
        cache = {}

    def dump():
        with open(file_name, 'w') as f:
            dill.dump(cache, f)

    # write cache to file at interpreter exit
    atexit.register(dump)
    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        def compute(key):
                cache[key] = func(*dill.loads(key[0]), **{k: v for k, v in key[1]})
                os.system('mkdir -p ' + os.path.dirname(file_name))

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

#def deep_frozenset(iterable):
#    if not isinstance(iterable, collections.Iterable):
#        return iterable
#    iterable = list(iterable)
#    for i, e in enumerate(iterable):
#        if isinstance(e, collections.Iterable):
#            iterable[i] = frozenset(map(deep_frozenset, e))
#    return frozenset(iterable)
#
#
#
## TODO: Should probably use atexit instead of reading/writing to disk at each
## involcation of the cached function, since this kills performance for functions
## that are called a large number of times.
## TODO: this is a good candidate for a utilities package
## TODO: use dill instead of frozenset throughout
## making arbitrary objects hashbable
#def persist_to_file(file_name, split = False):
#    """
#    Decorator for memoizing function calls to disk
#
#    Inputs:
#        file_name: File name prefix for the cache file(s)
#        split: If true, store each invocation of the function in a separate
#            file. This is useful for functions with large outputs that will
#            be called on a large numer of different input values.
#    """
#    # If we're not splitting each cached result into a single file then open
#    # the cache here.
#    if not split:
#        try:
#            with open(file_name, 'r') as f:
#                cache = dill.load(f)
#        except (IOError, ValueError):
#            cache = {}
#            print "no cache found"
#    else:
#        cache = {}
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
#
#        def new_func(*args, **kwargs):
#            # If we are splitting into multiple files, we can't retrieve the
#            # cache until here
#            if split:
#                full_name = file_name + '_' + str(hash(dill.dumps(args)))
#                try:
#                    with open(full_name, 'r') as f:
#                        new_cache = dill.load(f)
#                        for k, v in new_cache.items():
#                            cache[k] = v
#                except (IOError, ValueError):
#                    print "no cache found"
#            # if the "flush" kwarg is passed, recompute regardless of whether
#            # the result is cached
#            if "flush" in kwargs.keys():
#                kwargs.pop("flush", None)
#                key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
#                # TODO: refactor
#                if split:
#                    recompute(key, cache, full_name)
#                else:
#                    recompute(key, cache, file_name)
#            key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
#            if key not in cache:
#                if split:
#                    recompute(key, cache, full_name)
#                else:
#                    recompute(key, cache, file_name)
#            return cache[key]
#        return new_func
#
#    return decorator

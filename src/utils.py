import os
import dill
import collections

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

# TODO: use atexit or not?
# TODO: this is a good candidate for a utilities package
# TODO: use dill instead of frozenset throughout
# making arbitrary objects hashbable
def persist_to_file(file_name):

    try:
        with open(file_name, 'r') as f:
            cache = dill.load(f)
    except (IOError, ValueError):
        cache = {}

    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        def recompute(key):
                cache[key] = func(*dill.loads(key[0]), **{k: v for k, v in key[1]})
                os.system('mkdir -p ' + os.path.dirname(file_name))
                with open(file_name, 'w') as f:
                    dill.dump(cache, f)

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
                recompute(key)
            key = (dill.dumps(args), frozenset(kwargs.items()), frozenset(closure_dict.items()))
            if key not in cache:
                recompute(key)
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

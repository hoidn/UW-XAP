# Author: O. Hoidn

import numpy as np
import os
import cPickle
import dill
import pkg_resources
from time import time
import ipdb
import config
import hashlib
import itertools
import database
import playback
import random
#from datetime import import datetime
#from atomicwrites import atomic_write
#import collections
#from atomicfile import AtomicFile
#from libtiff import TIFF

PKG_NAME = __name__.split('.')[0]

def identity(x, **kwargs):
    return x

def random_float():
    random.seed(datetime.now())
    return random.uniform(0., 1.)

def merge_dicts(*args):
    final = {}
    for d in args:
        final.update(d)
    return final

def roundrobin(*iterables):
    """Merges iterables in an interleaved fashion.

    roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    # Recipe credited to George Sakkis
    if not iterables:
        raise ValueError("Arguments must be 1 or more iterables")
    nexts = itertools.cycle(iter(it).next for it in iterables)
    stopcount = 0
    while 1:
        try:
            for i, next in enumerate(nexts):
                yield next()
                stopcount = 0
        except StopIteration:
            stopcount += 1
            if stopcount >= len(iterables):
                break

def mpimap(func, lst):
    """
    Map func over list in parallel over all MPI cores.

    The full result is returned in each rank.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    results = \
        [func(elt)
        for n, elt in enumerate(lst)
        if n % size == rank]
    results = comm.allgather(results)
    if results:
        results = list(roundrobin(*results))
    return results

def isroot():
    """
    Return true if the MPI core rank is 0 and false otherwise.
    """
    if 'OMPI' not in ' '.join(os.environ.keys()):
        return True
    else:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        return (rank == 0)
    
def ifroot(func):
    """
    Decorator that causes the decorated function to execute only if
    the MPI core rank is 0.
    """
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
    @ifroot
    def inner(*args, **kwargs):
        import config
        if config.noplot:
            print "PLOTTING DISABLED, EXITING." 
        else:
            return func(*args, **kwargs)
    return inner
    
        
@playback.db_insert
@ifroot
def save_image(save_path, imarr, fmt = 'tiff'):
    """
    Save a 2d array to file as an image.
    """
    if not isinstance(imarr, np.ndarray):
        imarr = np.array(imarr)
    import Image
    from scipy import misc
    import matplotlib
    dirname = os.path.dirname(save_path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(save_path))
    np.save(save_path + '.npy', imarr)
    if imarr.dtype == 'uint16':
        imarr = imarr.astype('float')
    im = Image.fromarray(imarr)
    im.save(save_path + '.tif')
    matplotlib.image.imsave(save_path + '.png', imarr)

@playback.db_insert
@ifroot
def save_data(x, y, save_path):
    dirname = os.path.dirname(save_path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(save_path))
    np.savetxt(save_path, [x, y])
    database.mongo_add(save_path, [list(x), list(y)])

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


def save_image_and_show(save_path, imarr, title = 'Image', rmin = None, rmax = None, show_plot = True):
    """
    Save a 2d array to file as an image and then display it.
    """
    ave, rms = imarr.mean(), imarr.std()
    if not rmin:
        rmin = ave - rms
    if not rmax:
        rmax = ave + 5 * rms
    @ifplot
    def show():
        print "rmin", rmin
        print "rmax", rmax
        import pyimgalgos.GlobalGraphics as gg
        gg.plotImageLarge(imarr, amp_range=(rmin, rmax), title = title)
        if show_plot:
            gg.show()
    save_image(save_path, imarr)
    show()


@playback.db_insert
@ifplot
def global_save_and_show(save_path):
    """
    Save current matplotlib plot to file and then show it.
    """
    import matplotlib.pyplot as plt
    dirname = os.path.dirname(save_path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.show()

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    import inspect
    args, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults:
        return dict(zip(args[-len(defaults):], defaults))
    else:
        return {}

def resource_f(fpath):
    from StringIO import StringIO
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
    return a hash of any python object
    """
    if isinstance(obj, str) or isinstance(obj, np.ndarray):
        return hashlib.sha1(dill.dumps(obj)).hexdigest()
    try:
        # This is a workaround for a cryptic error that occrs in pickle when
        # obj is an iterable type (specifically: 'I/O operation on closed file')
        slist =\
            [make_hashable(x)
            for x in obj]
        return hashlib.sha1(''.join(slist)).hexdigest()
    except TypeError:
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

def memoize_condition(cache_valid):
    """
    Memoization operator that invalidates cache whenever cache_valid()
    evaluates to False.
    """
    cache = {}
    def decorator(f):
        def new_func(*args):
            if (args in cache) and cache_valid():
                return cache[args]
            else:
                cache[args] = f(*args)
                return cache[args]
        return new_func
    return decorator

def memoize_timeout(timeout = 10):
    state = {}
    def cache_valid():
        if 'last' not in state:
            state['last'] = time()
        curtime = time()
        if curtime - state['last'] > timeout:
            state['last'] = curtime
            return False
        else:
            return True
    return memoize_condition(cache_valid)

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
                    #print "no cache file found"
                    pass
                flag_cache_loaded()
            if not key in cache.keys():
                cache[key] = func(*dill.loads(key[0]), **{k: v for k, v in key[1]})
                if not check_cache_changed():
                    # write cache to file at interpreter exit if it has been
                    # altered
                    import atexit
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

def eager_persist_to_file(file_name, excluded = None, rootonly = True):
    """
    Decorator for memoizing function calls to disk.
    Differs from persist_to_file in that the cache file is accessed and updated
    at every call, and that each call is cached in a separate file. This allows
    parallelization without problems of concurrency of the memoization cache,
    provided that the decorated function is expensive enough that the
    additional read/write operations have a negligible impact on performance.

    Inputs:
        file_name: File name prefix for the cache file(s)
        rootonly : boolean
                If true, caching is only applied for the MPI process of rank 0.
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
            # TODO: merged_dict: why aren't changes in kwargs reflected in it?
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
            key = make_hashable(tuple(map(make_hashable, [args, merged_dict, closure_dict.items(), list(kwargs.iteritems())])))
            #print "key is", key
#            for k, v in kwargs.iteritems():
#                print k, v
            return key

        @ifroot# TODO: fix this
        def dump_to_file(d, file_name):
            os.system('mkdir -p ' + os.path.dirname(file_name))
            with open(file_name, 'w') as f:
                cPickle.dump(d, f)
            #print "Dumped cache to file"
    
        def compute(*args, **kwargs):
            file_name = kwargs.pop('file_name', None)
            key = gen_key(*args, **kwargs)
            value = func(*args, **kwargs)
            cache[key] = value
            # Write to disk if the cache file doesn't already exist
            if not os.path.isfile(file_name):
                dump_to_file(value, file_name)
            return value

        def new_func(*args, **kwargs):
            # Because we're splitting into multiple files, we can't retrieve the
            # cache until here
            #print "entering ", func.func_name
            key = gen_key(*args, **kwargs)
            full_name = file_name + key
            if key not in cache:
                try:
                    try:
                        with open(full_name, 'r') as f:
                            cache[key] = cPickle.load(f)
                    except EOFError:
                        os.remove(full_name)
                        print "corrupt cache file deleted"
                        raise ValueError("Corrupt file")
                    #print "cache found"
                except (IOError, ValueError):
                    #print "no cache found; computing"
                    compute(*args, file_name = full_name, **kwargs)
            # if the "flush" kwarg is passed, recompute regardless of whether
            # the result is cached
            if "flush" in kwargs.keys():
                kwargs.pop("flush", None)
                # TODO: refactor
                compute(*args, file_name = full_name, **kwargs)
            #print "returning from ", func.func_name
            return cache[key]

        return new_func

    return decorator

@eager_persist_to_file("cache/xrd.combine_masks/")
def combine_masks(imarray, mask_paths, verbose = False, transpose = False):
    """
    Takes a list of paths to .npy mask files and returns a numpy array
    consisting of those masks ANDed together.
    """
    # Initialize the mask based on zero values in imarray.
    import numpy.ma as ma
    base_mask = ma.make_mask(np.ones(np.shape(imarray)))
    base_mask[imarray == 0.] = False
    if not mask_paths:
        print "No additional masks provided"
        return base_mask
    else:
        # Data arrays must be transposed here for the same reason that they
        # are in data_extractor.
        if transpose:
            masks = map(lambda path: np.load(path).T, mask_paths)
        else:
            masks = map(lambda path: np.load(path), mask_paths)
        print "Applying mask(s): ", mask_paths
        return base_mask & reduce(lambda x, y: x & y, masks)


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


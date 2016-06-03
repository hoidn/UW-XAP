"""
Functions for controlling logging and other output.
"""

from __future__ import print_function
import os
import config
import logging

logging.basicConfig(filename = config.logfile_path, level = logging.DEBUG)

class conditional_decorator(object):
    """
    From http://stackoverflow.com/questions/10724854/how-to-do-a-conditional-decorator-in-python-2-6
    """
    def __init__(self, dec, condition):
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)

def isroot():
    """
    Return true if the MPI core rank is 0 and false otherwise.
    """
    if 'OMPI_COMM_WORLD_RANK' not in ' '.join(os.environ.keys()):
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

def stdout_to_file(path = None):
    """
    Decorator that causes stdout to be redirected to a text file during the
    modified function's invocation.
    """
    if path is None:
        path = 'mecana.log'
    def decorator(func):
        import sys
        def new_func(*args, **kwargs):
            stdout = sys.stdout
            sys.stdout = open(path, 'a')
            result = func(*args, **kwargs)
            sys.stdout.close()
            sys.stdout = stdout
            return result
        return new_func
    return decorator

#@conditional_decorator(stdout_to_file(path = config.logfile_path), config.stdout_to_file)
@conditional_decorator(ifroot, config.suppress_root_print)
def log(*args):
    def newargs():
        if config.stdout_to_file:
            return ('PID ', os.getpid(), ': ') + args
        else:
            return args
    logging.info(' '.join(map(str, args)))
    #print(*newargs())

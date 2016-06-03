# Author: O. Hoidn
import re
import os
import inspect
import time
from output import log
# relative path of autoscripts directory
AUTOSCRIPT_DIR = 'autoscripts/'

EXTRA_IMPORTS =\
"""
import sys
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/pathos-0.2a1.dev0-py2.7.egg')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dataccess-1.0-py2.7.egg')
"""


def makescript(sourcefile, target_command, cache_path, mode = 'interactive'):
    """
    Decorator that replaces each call to the decorated function with a script
    (saved to the directory autoscripts) that takes no arguments and,
    when run, evaluates that original function call as its last expression.

    Note that all arguments to the decorated function must be simple types.
    More specifically: for a variable x, it is necessary that eval(repr(x)) == x.

    Parameters
    ---------
    sourcefile : string
        path of the python module from which the script is generated
    target_command : format string
        format string that converts the path of a script file to a shell command
        to run.
    cache_path : str
        A path prefix for caching return values using utils.persist_to_file.
    mode : str
        either 'interactive' or 'batch'.
    """
    if sourcefile[-1] == 'c': # source is a pyc file
        sourcefile = sourcefile[:-1]

    def decorator_interactive(f):
        def passthrough(*args, **kwargs):
            return f(*args, **kwargs)
        return passthrough

    def decorator_script(f):
        if not os.path.exists(AUTOSCRIPT_DIR):
            os.mkdir(AUTOSCRIPT_DIR)
        # get name of the function
        name = f.__name__ 
        # strip decorators from the function's source code and de-indent it
        sourcelist = [line[4:] for line in inspect.getsource(f).split('\n')
            if (line and '@' not in line)]
        f_code = '\n'.join(sourcelist)
        log( f_code)
        with open(sourcefile, 'r') as sourcef:
            file_code = sourcef.readlines()
            #print file_code
        decorator_line = "\n@utils.eager_persist_to_file('" + cache_path + "')"
        # strip this decorator from the script code
        filtered_file_code = EXTRA_IMPORTS + ''.join(filter(lambda l: 'makescript' not in l, file_code)) + '\n' + decorator_line + '\n' + f_code 
        def g(*args, **kwargs):
            argstring = ', '.join(map(repr, args))
            kwargstrings = [k + ' = ' + repr(v) for k, v in kwargs.iteritems()]
            kwargstring = ', '.join(kwargstrings)
            autoscript_name = AUTOSCRIPT_DIR + f.__name__ + '_' + '_'.join(map(str, args)) + '_'.join(map(str, kwargs.values())) + '.py'
            if kwargstring:
                funcall_string = name + '(' + argstring + ', ' + kwargstring + ')'
            else:
                funcall_string = name + '(' + argstring + ')'
            with open(autoscript_name, 'w') as script:
                script.write(filtered_file_code + '\n' + funcall_string)
            log( target_command % autoscript_name)
            return target_command % autoscript_name
#            time.sleep(1)
#            os.system(target_command % autoscript_name)
        return g

    if mode == 'interactive':
        return decorator_interactive
    elif mode == 'script':
        return decorator_script
    else:
        raise KeyError("'mode': invalid value")


# Author: O. Hoidn

import re
import os




def makescript(sourcefile):
    """
    Decorator that replaces each call to the decorated function with a script
    (saved to the directory autoscripts) that takes no arguments and,
    when run, evaluates that original function call as its last expression.

    Note that all arguments to the decorated function must be simple types.
    More specifically: for a variable x, it is necessary that eval(repr(x)) == x.
    """
    if sourcefile[-1] == 'c': # source is a pyc file
        sourcefile = sourcefile[:-1]
    def decorator(f):
        if not os.path.exists('autoscripts/'):
            os.mkdir('autoscripts')
        # get name of the function
        name = f.__name__ 
        with open(sourcefile, 'r') as f:
            file_code = f.readlines()
            print file_code
        # strip this decorator from the script code
        filtered_file_code = ''.join(filter(lambda l: 'makescript' not in l, file_code))
        def g(*args, **kwargs):
            argstring = ', '.join(map(repr, args))
            kwargstrings = [k + ' = ' + repr(v) for k, v in kwargs.iteritems()]
            kwargstring = ', '.join(kwargstrings)
            if kwargstring:
                funcall_string = name + '(' + argstring + ', ' + kwargstring + ')'
            else:
                funcall_string = name + '(' + argstring + ')'
#            print funcall_string
#            print filtered_file_code
#            print sourcefile
            with open('autoscripts/' + __name__ + '_' + '_'.join(map(str, args)) + '.py', 'w') as f:
                f.write(filtered_file_code + '\n' + funcall_string)
        return g
    return decorator

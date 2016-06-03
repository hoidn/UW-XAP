import dill
import os
import hashlib
from output import log

# TODO: fix this module

"""
Module for storing and replaying individual function calls without re-running
any external code, including that which created a function's evaluation environment.
This is implemented thanks to dill's ability to store an entire interpreter session.
"""


db_dir = 'db/'
db = []


def hash(obj):
    """
    return hash of an object that supports python's buffer protocol
    """
    return hashlib.sha1(dill.dumps(obj)).hexdigest()

def get_fname(key):
    return key.strip('/')

def load_db(key):
    fname = get_fname(key)
    try:
        with open(db_dir + fname, 'r') as f:
            local = dill.load(f)
            for func in local:
                db.append(func)
    except IOError:
        raise KeyError("key not found")

def save_db(key):
    """
    Saves all stored calls to functions
    decorted by db_insert to disk. This function complements
    load_db().
    """
    fname = get_fname(key)
    if not os.path.exists(db_dir):
        os.system('mkdir -p %s' % db_dir)
    with open(db_dir + fname, 'w') as f:
        dill.dump(db, f)
    log( "saved db")

def execute():
    """
    Evaluate all fuction calls that have been stored by db_insert()
    """
    while db:
        db.pop(0)()

def db_insert(func):
    """
    Decorator that causes the modified function call to be stored instead of
    evaluated.
    """
    import config
    if not config.playback:
        return func
    else:
        def inner(*args, **kwargs):
            def execute():
                return func(*args, **kwargs)
            db.append(execute)
        return inner

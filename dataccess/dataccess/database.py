import hashlib
import ipdb
import dill
import os

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
    #fname = hash(dill.dumps(key))
    fname = get_fname(key)
    try:
        with open(db_dir + fname, 'r') as f:
            local = dill.load(f)
            for func in local:
                db.append(func)
    except IOError:
        raise KeyError("key not found")

def save_db(key):
    #fname = hash(dill.dumps(key))
    fname = get_fname(key)
    if not os.path.exists(db_dir):
        os.system('mkdir -p %s' % db_dir)
    with open(db_dir + fname, 'w') as f:
        dill.dump(db, f)
    print "saved db"

def execute():
    while db:
        db.pop(0)()

def db_insert(func):
    def inner(*args, **kwargs):
        def execute():
            return func(*args, **kwargs)
        db.append(execute)
    return inner

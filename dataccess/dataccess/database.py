import hashlib
import ipdb
import dill
import os
import hashlib
import config
from pymongo import MongoClient

"""
Module for storing and replaying individual function calls within mecana's modules,
and for storing outputs from mecana in a MongoDB db.
"""

MONGO_HOST = 'pslogin03'
MONGO_PORT = 5837

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


# Functions for interacting with Mongo and inserting into/loading from
# appending to the interpreter-wide Mongodb cache.
to_insert = {}
client = MongoClient(MONGO_HOST, MONGO_PORT)
collection = client.database[config.expname]

# Value of name field for google spreadsheet logbook document
# pseudo-random string of reasonable length that shouldn't collide with other keys
logbook_name = hash('name') 

def mongo_init(key):
    to_insert['key'] = key

def hash(obj):
    return hashlib.sha1(dill.dumps(obj)).hexdigest()

def get_state_hash(dependency_dicts):
    """
    Return a string containing the hash of each element in a list.
    """
    return '_'.join(map(hash, dependency_dicts))


def mongo_add(key, obj):
    """
    Add a key to the global to_insert dict, but don't yet insert
    the dict into the database.
    """
    to_insert[key] = obj

#def mongo_load(

def mongo_insert_logbook_dict(d):
    d['name'] = logbook_name
    inserted = collection.insert(d, check_keys = False)
    if list(collection.find({'_id': {"$ne": inserted}})):
        collection.remove({'_id': {"$ne": inserted}})
        print 'removed'

def mongo_get_logbook_dict():
    raw_dict = list(collection.find({"name": logbook_name}))[0]
    for k, v in raw_dict.iteritems():
        if isinstance(v, dict) and 'runs' in v:
            v['runs'] = tuple(v['runs'])
    return {k: v for k, v in raw_dict.iteritems() if isinstance(v, dict)}

def mongo_commit(label_dependencies = None):
    from dataccess import logbook
    spreadsheet = logbook.get_pub_logbook_dict()
    dependency_dicts =\
        [spreadsheet[label]
        for label in label_dependencies]
    try:
        key = to_insert['key']
        state_hash = get_state_hash(dependency_dicts)
    except KeyError:
        raise KeyError("Attempting to insert non-initialized dict into mongo database")
    if not list(collection.find({'key': key, 'state_hash': state_hash})):
        to_insert['state_hash'] = state_hash
        collection.insert_one(to_insert) 

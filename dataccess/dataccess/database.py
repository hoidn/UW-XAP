import hashlib
import ipdb
import dill
import os
import hashlib
import config
from pymongo import MongoClient
import cPickle
import gridfs

"""
Interface module for mecana's MongoDB collection.

Includes functions for storing and replaying individual function calls within 
mecana's modules, for storing logging spreadsheet data, for caching mecana's
outputs, and for inserting and accessing derived datasets.
"""

MONGO_HOST = 'pslogin03'
MONGO_PORT = 4040

db_dir = 'db/'
db = []


def hash(obj):
    """
    return hash of an object that supports python's buffer protocol
    """
    return hashlib.sha1(dill.dumps(obj)).hexdigest()

def get_fname(key):
    return key.strip('/')

# TODO: Move this data from the filesystem to MongoDB, if it's straighforward to do so
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
# Use GridFS to fit store objects > 16 MB
FS = gridfs.GridFS(client.database)



def mongo_init(key):
    to_insert['key'] = key

def hash(obj):
    return hashlib.sha1(dill.dumps(obj)).hexdigest()



def mongo_add(key, obj):
    """
    Add a key to the global to_insert dict, but don't yet insert
    the dict into the database.
    """
    to_insert[key] = obj

def mongo_insert_logbook_dict(d):
    """
    Insert logging spreadsheet data into MongoDB.
    """
    # Set the value of the name field to something unique for each google spreadsheet
    d['name'] = config.logbook_ID
    inserted = collection.insert(d, check_keys = False)
    if list(collection.find({'_id': {"$ne": inserted}, 'name': {"$eq": config.logbook_ID}})):
        collection.remove({'_id': {"$ne": inserted}, 'name': {"$eq": config.logbook_ID}})
        print 'removed'

def mongo_get_logbook_dict():
    """
    Return the logging spreadsheet data dictionary.
    """
    raw_dict = list(collection.find({"name": config.logbook_ID}))[0]
    for k, v in raw_dict.iteritems():
        if isinstance(v, dict) and 'runs' in v:
            v['runs'] = tuple(v['runs'])
    return {k: v for k, v in raw_dict.iteritems() if isinstance(v, dict)}

def mongo_commit(label_dependencies = None):
    """
    Insert the interpreter session's cache (created by calls to mongo_add)
    into MongoDB.
    """
    def get_state_hash(dependency_dicts):
        """
        Return a string containing the hash of each element in a list.
        """
        return '_'.join(map(hash, dependency_dicts))
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
        collection.insert(to_insert, check_keys = False) 


def mongo_insert_derived_dataset(data_dict):
    """
    Insert query output data into MongoDB.

    data_dict : dict
        This dict must contain the following keys/value at a minimum:
        -'label': The label to assign to this dataset.
        -'detid', ID of the dataset's detector.
        -'data', a tuple containing the query's averaged detector readout and
        event data dictionary.
        -All logbook attributes that were used to evaluate the query.
    """
    # serialize the dict
    blob = cPickle.dumps(data_dict)
    to_insert = {'gridFS_ID': FS.put(blob)}
    to_insert['label'] = data_dict['label']
    to_insert['detid'] = data_dict['detid']
    to_insert['source_logbook'] = config.logbook_ID
    collection.insert(to_insert)

def mongo_query_derived_dataset(label, detid):
    """
    Return a query output dataset previously inserted by mongo_insert_derived_dataset.
    """
    result_list =\
        list(collection.find({'source_logbook': config.logbook_ID,
            'label': {'$regex': label}, 'detid': detid}))
    if not result_list:
        return None
    result = result_list[0]
    if len(result_list) > 1:
        print "WARNING: regex '%s' matches more than one derived dataset. First match will be selected: %s" % (label, result['label'])
    blob = FS.get(result['gridFS_ID']).read()
    return cPickle.loads(blob)

import hashlib
import ipdb
import dill
import config
from pymongo import MongoClient
import cPickle
import gridfs
import os
import binascii
import utils

"""
Interface module for mecana's MongoDB collection.

Includes functions for storing and replaying individual function calls within 
mecana's modules, for storing logging spreadsheet data, for caching mecana's
outputs, and for inserting and accessing derived datasets.
"""

MONGO_HOST = 'pslogin03'
MONGO_PORT = 4040
token = ''

db_dir = 'db/'
db = []


def hash(obj):
    """
    return hash of an object that supports python's buffer protocol
    """
    return hashlib.sha1(dill.dumps(obj)).hexdigest()

# Functions for interacting with Mongo and inserting into/loading from
# appending to the interpreter-wide Mongodb cache.
to_insert = {}
client = MongoClient(MONGO_HOST, MONGO_PORT)
collection = client.database[config.expname + token]
# Use GridFS to fit store objects > 16 MB
FS = gridfs.GridFS(client.database)

def dumps_b2a(obj):
    """
    Convert an object into an ASCII string that can be inserted into
    MongoDB.
    """
    return binascii.b2a_base64(dill.dumps(obj))

def loads_a2b(ascii_str):
    """
    Decode an object that was encoded by dumps_b2a.
    """
    return dill.loads(binascii.a2b_base64(ascii_str))

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

@utils.ifroot
def mongo_replace(collection, d, mongo_query_dict):
    """
    mongo_query_dict: a query that will match stale documents
    that must be removed.
    """
    remove_query_dict = {k: v for k, v in mongo_query_dict.iteritems()}
    inserted = collection.insert(d, check_keys = False)
    remove_query_dict['_id'] = {"$ne": inserted}
    if list(collection.find(remove_query_dict)):
        collection.remove(remove_query_dict)
        print "removed"


def mongo_insert_logbook_dict(d):
    """ 
    Insert logging spreadsheet data into MongoDB.
    """
    d['name'] = config.logbook_ID
    collection = client.database[config.expname + token]
    query_dict = {'name': {"$eq": config.logbook_ID}}
    mongo_replace(collection, d, query_dict)

def mongo_get_logbook_dict():
    """
    Return the logging spreadsheet data dictionary.
    """
    raw_dict = list(collection.find({"name": config.logbook_ID}))[0]
    for k, v in raw_dict.iteritems():
        if isinstance(v, dict) and 'runs' in v:
            v['runs'] = tuple(v['runs'])
    return {k: v for k, v in raw_dict.iteritems() if isinstance(v, dict)}

@utils.ifroot
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

    dependency_dicts =\
        [logbook.get_label_dict(label)
        for label in label_dependencies]
    try:
        key = to_insert['key']
        state_hash = get_state_hash(dependency_dicts)
    except KeyError:
        raise KeyError("Attempting to insert non-initialized dict into mongo database")
    if not list(collection.find({'key': key, 'state_hash': state_hash})):
        to_insert['state_hash'] = state_hash
        collection.insert(to_insert, check_keys = False) 

def mongo_find(key):
    return list(collection.find({'key': key}))

def mongo_store_object_by_label(obj, label):
    """
    Store a python object to MongoDB.
    """
    collection = client.database[config.logbook_ID + '_objects_by_label']
    d = {'label': label, 'object': dumps_b2a(obj)}
    query_dict = {'label': label}
    mongo_replace(collection, d, query_dict)

def mongo_query_object_by_label(label):
    """
    Query a python object stored to MongoDB.
    """
    collection = client.database[config.logbook_ID + '_objects_by_label']
    result_list = list(collection.find({'label': label}))
    if not result_list:
        raise KeyError("%s: object not found" % label)
    # TODO: treat case of multiple results
    return loads_a2b(result_list[0]['object'])
    
    
# TODO: move all these functions into a module or class
@utils.ifroot
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
    collection = client.database[config.expname + token + '_derived']
    # initialize to_insert with the remaining key/value pairs. These include
    # all applicable logbook attributes.
    to_insert =\
        {k: v
        for k, v in data_dict.iteritems()
        if k != 'data'}
    try:
        # extract the (frame, event data dict) tuple
        blob = cPickle.dumps(data_dict.pop('data'))
        # Serialize the data tuple and store it
        to_insert['gridFS_ID'] = FS.put(blob)

        to_insert['detid'] = data_dict['detid']
        to_insert['event_data_getter'] = data_dict['event_data_getter']
    except KeyError:
        pass
    to_insert['label'] = data_dict['label']
    to_insert['source_logbook'] = config.logbook_ID
    collection.insert(to_insert)


def mongo_get_all_derived_datasets():
    """
    Return a dictionary in the same format as that returned by logbook.get_pub_logbook_dict().
    """
    collection = client.database[config.expname + token + '_derived']
    documents =\
        list(collection.find({'source_logbook': config.logbook_ID}))
    def insert_one(d):
        label = d.pop('label')
        return label, d
    attribute_dict =\
        {k: v
        for k, v in map(insert_one, documents)}
    return attribute_dict
    

def mongo_query_derived_dataset(label, detid, event_data_getter = None):
    """
    Return a query output dataset previously inserted by mongo_insert_derived_dataset.
    
    The return value is a tuple containing an averaged frame and an event data dictionary.
    """
    collection = client.database[config.expname + token + '_derived']
    result_list =\
        list(collection.find({'source_logbook': config.logbook_ID,
            'label': {'$regex': label}, 'detid': detid, 'event_data_getter': dumps_b2a(event_data_getter)}))
    if not result_list:
        dataset = mongo_query_object_by_label(label)
        return dataset.evaluate(detid, event_data_getter = event_data_getter)
    result = result_list[0]
    if len(result_list) > 1:
        print "WARNING: regex '%s' matches more than one derived dataset. First match will be selected: %s" % (label, result['label'])
    blob = FS.get(result['gridFS_ID']).read()
    print "loading dataset from MongoDB"
    return cPickle.loads(blob)


def get_derived_dataset_attribute(pat, attribute):
    """
    Return the value of an attribute belonging to a derived dataset.

    Raises a KeyError if the label isn't found
    """
    import re
    pat = '^' + pat + '$'
    attribute_map = mongo_get_all_derived_datasets()
    matching_labels =\
        filter(lambda lab: bool(re.search(pat, lab, flags = re.IGNORECASE)),
            attribute_map.keys())
    if not matching_labels:
        raise KeyError("%s: no matching derived dataset label found" % pat) 
    result_label = matching_labels[0]
    if len(matching_labels) > 1:
        print "WARNING: regex '%s' matches more than one derived dataset. First match will be selected: %s" % (pat, result_label)
    return attribute_map[result_label][attribute]

def delete_collections():
    # TODO: flush cache in data_access as well
    collections =\
        [client.database[config.expname + token],
        client.database[config.expname + token + '_derived'],
        client.database[config.logbook_ID + '_objects_by_label']]
    for collection in collections:
        collection.delete_many({})
    os.system('rm -rf cache/query/DataSet.evaluate*')

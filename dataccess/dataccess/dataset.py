import copy
import config
import utils
import dataquery
import database
import logbook
import psana_get

"""
Module for accessing datasets by label.
"""

    
#def serve_dataset_query(detid, runs, event_filter, event_data_getter):
#    """
#    Arguments:
#    detid : str
#        detector ID
#    runs : list of ints
#        run numbers
#    event_filter : EventFilter
#        specifies events to accept.
#    event_data_getter : function
#        Function with which to compute event-by-event derived data.
#    Returns a DatasetQueryResult instance.
#    """
    

class DatasetStore(object):
    mongo_collection = database.get_collection('DatasetStore')
    def insert_dataset(self, ds):
        mongo_collection.insert({'label': ds.label, 'object': database.dumps_b2a(ds)}, check_keys = False)
    def find_database(self, label):
        """
        Attempts to parse label as a run range specifier if no match is found in the global
        collection of datasets. If the label is a valid run range specifier, the corresponding
        dataset is added to the global collection.
        """
        def mongo_lookup():
            result_list = list(mongo_collection.find({'label': label}))
            if not result_list: 
                raise KeyError("%s: object not found" % label)
            return database.loads_a2b(result[0]['object'])
        try:
            dataset = mongo_lookup()
        except KeyError, e1: # try to parse the label as range of runs and then look for matches
            try:
                dataset = copy.deepcopy(runs_to_dataset(logbook.parse_run(label)))
                dataset.runs = tuple(runset)
                dataset_store.insert_dataset(dataset)
            except ValueError, e2:
                raise ValueError(str(e1) + '; ' + str(e2))
        return dataset
    def get_dataset_attribute_value(self, label, attribute):
        dataset = self.find_database(label)
        return dataset.get_attribute_value(attribute)
    def dump_all_datasets(self):
        return list(mongo_collection.find())

dataset_store = DatasetStore()

EventFilter = namedtuple('EventFilter', ['filter_function', 'filter_detid'])
#class EventFilter(object):
#    """
#    Class to store an event filter, consisting of a function and a detector ID
#    to whose data the function is to be applied.
#    """
#    def __init__(self, filter_function, filter_detid):
#        assert hasattr(filter_function, '__call__')
#        assert isinstance(filter_detid, str)
#        self.filter_function = filter_function
#        self.filter_detid = filter_detid

class DataSet(object):
    def __init__(self, runs, attribute_dict, label, darkruns = None, event_filter = None):
        """
        Instantiate a dataset label and register it into the database.

        runs : list
            A list of run numbers
        event_filter : EventFilter
            An EventFilter instance specifiying which events are accepted and
            rejected.
        label : string
            Dataset label, to be used to query this dataset from Database.query_data
        """
        assert isinstance(attribute_dict, dict)
        self.runs = runs
        self.runs.sort()
        self.darkruns = darkruns
        self.event_filter = event_filter
        self.label = label
        self.attribute_dict = attribute_dict
        self.query_cache = {}
        dataset_store.insert_dataset(self)

#    def db_insert(self):
#        self.dataset_store.insert(self, label = self.label)

    @utils.eager_persist_to_file('cache/dataquery/DataSet/')
    def query_data(self, detid, event_data_getter = None):
        """
        Query data belonging to/derived from this object.
        """
        return psana_get.get_signal_many_parallel_filtered(self.event_filter,
            self.runs, detid, event_data_getter = event_data_getter,
            darkruns = self.darkruns)

    def get_attribute_value(self, attr):
        return self.attribute_dict[attr]

def runs_to_dataset(runlist):
    """
    Given a list/tuple of run numbers, return the matching DataSet from the
    global collection.
    """
    runset = set(runlist)
    datasets = dataset_store.dump_all_datasets()
    for ds in datasets:
        if runset <= set(ds.runs):
            return dataset
    raise KeyError("No matching dataset found.")

def get_dataset_query_result(label, detid, event_data_getter = None):
    """
    Return a dataset's data for the specified detector ID and derived data evaluator.
    """
    dataset = dataset_store.find_database(label)
    return datset.query_data(detid, event_data_getter = event_data_getter)

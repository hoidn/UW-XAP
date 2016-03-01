"""
For querying datasets by spreadsheet attribute values.
"""

import sys
import re
import ipdb
import config
import numpy as np
import matplotlib.pyplot as plt
import logbook
import utils
import summarymetrics
import data_access
import database
from recordclass import recordclass



def query_generic(attribute, value_test_func):
    """
    Given a dataset attribute and function that, given values of
    that attribute, returns a boolean, return a set of all matching
    run numbers.
    """
    def new_value_test_func(val):
        try:
            return value_test_func(val)
        except:
            return False
    def test(row):
        try:
            return new_value_test_func(row[attribute])
        except KeyError:
            return False
    relational = []
    d = logbook.get_pub_logbook_dict()
    for k, v in d.iteritems():
        v['label'] = k
        relational.append(v)
    result =\
        [row['runs']
        for row in relational
        if np.all(test(row))]
    if not result:
        return set()
    else:
        s = set(reduce(lambda x, y: x + y, filter(lambda t: t != (None,), result)))
        return set(filter(lambda x: x is not None, s))

# Named tuple definition for a query.
Query = recordclass('Query', ['attribute', 'function', 'attribute_value', 'label'])

def construct_query(attribute, param1, param2 = None):
    """
    Create a single Query record with a matching function based on
    regex search and numeric comparison for string and numeric types,
    respectively.

    Arguments:
    attribute : str
        A logbook attribute
    param1 : str or numeric
        Type must correspond to that of attribute's cell values.
    param2 : numeric
        Only for attributes with numeric values. If not provided, the query
        will check for exact equality of cell values with param1. If provided,
        param1 and param2 define the range of accepted cell values.
    Returns a Query instance
    """
    # TODO: Validate data type of param1 and param2 and raise exception if it's wrong.
    def match_string():
        bash_special = r"|&;<>()$`\"' \t\n"
        pat = param1.encode('string-escape')
        pat_bashsafe = filter(lambda c: c not in bash_special, pat)
        label = '-'.join(map(str, [attribute, param1]))
        def match(cell_value):
            return bool(re.search(pat, cell_value, flags = re.IGNORECASE))
        return Query(attribute, match, pat_bashsafe, label)
    def match_numeric():
        if param2 is None:
            label = '-'.join(map(str, [attribute, param1]))
            def matchsingle(cell_value):
                return cell_value == param1
            return Query(attribute, matchsingle, param1, label)
        else:
            label = '-'.join(map(str, [attribute, param1, param2]))
            def matchrange(cell_value):
                # Assume the cell value is either numeric, or an iterable containing only numerics
                return (param1 <= np.min(cell_value)) and (cell_value <= np.max(param2))
            return Query(attribute, matchrange, None, label)
    if isinstance(param1, str):
        return match_string()
    else:# param1 is numeric
        return match_numeric()
        
# TODO: tests
class DataSet(object):
    """
    A class representing a set of events corresponding to the intersection of
    (1) a run query and (2) events for which event_filter returns True.

    An empty run query matches all runs in the logging spreadsheet.

    Public methods:
        -evaluate(): extracts data from a DataSet instance.
    """
    def __init__(self, query, event_filter = None, event_filter_detid = None,
            label = None, immediate_insert = True):
        """
        query : list
            A list of Query records.
        """
        if event_filter and not event_filter_detid:
            raise ValueError("event_filter_detid must be provided if event_filter is not None")
        assert (event_filter is None) or hasattr(event_filter, '__call__')
        if not query or len(query) == 0:
            query = [construct_query('label', '.*')]
        self.query = query
        runsets =\
            [query_generic(q.attribute, q.function)
            for q in query]
        self.runs = list(reduce(lambda x, y: x & y, runsets))
        self.runs.sort()
        print "Matching runs: %s" % str(self.runs)
        self.event_filter = event_filter
        self.event_filter_detid = event_filter_detid
        if label is None:
            query_strings = [q.label for q in query]
#            attribute_pairs =\
#                [q.attribute + '-' + str(q.attribute_value)
#                for q in query
#                if q.attribute_value is not None]
            if event_filter is not None:
                try:
                    filter_identifier = '-'.join(event_filter.params)
                except:
                    filter_identifier = database.hash(utils.random_float())
                filter_label = '-filter-' + event_filter.__name__ + '-' + filter_identifier + '-' + str(event_filter_detid)
                #filter_label = '-filter-' + event_filter.__name__ + '-' + database.hash(event_filter) + '-' + event_filter_detid
            else:
                filter_label = ''
            self.label = '-'.join(query_strings) + filter_label
        else:
            self.label = label
        # Store this entire data structure
        self._store() 
        # Store a dict mapping self.label to the query parameters
        self._db_insert() 
        # TODO: Have a way to insert a DataSet into MongoDB without having to do this
        # type of dummy evaluation.
#        if immediate_insert:
#            self.evaluate('GMD')

    def _store(self):
        """
        Store this object to MongoDB.
        """
        database.mongo_store_object_by_label(self, self.label)

    @utils.eager_persist_to_file('cache/query/DataSet.evaluate')
    def evaluate(self, detid, event_data_getter = None, insert = True):
        """
        Extracts data for all runs in self.run, subject to the specified event-by-event
        filtering.

        Returns:
        make_frame : 2d np.ndarray
            The mean area detector readout over all events (or all events accepted
            by event_filter)
        merged : dict
            The event data dictionary for all events (or all events accepted by event_filter)
        """
        if event_data_getter is None:
            def eventsum(arr, **kwargs):
                return np.sum(arr)
            # so that we can get the number of events per run even if event_data_getter is None            
            # TODO: this is a temporary solution, find a better one.
            event_data_getter = eventsum
        runs = self.runs
        labels = map(str, runs)
        pairs =\
            [data_access.get_data_and_filter(label, detid,
                event_data_getter = event_data_getter, event_filter = self.event_filter,
                event_filter_detid = self.event_filter_detid)
                for label in labels]
        frames = np.array(map(lambda x: x[0], pairs))
        event_data_dicts = map(lambda x: x[1], pairs)
        def dict_nevents(d):
            return len(d.values()[0].keys())
        nevents_per_run = np.array(map(dict_nevents, event_data_dicts))

        merged = utils.merge_dicts(*event_data_dicts)
        # Mean of the averaged frames, weighted by the number of events
        # processed from each run.
        mean_frame = reduce(lambda x, y: x + y,
            nevents_per_run[:, None, None] * frames)/np.sum(nevents_per_run)
        if insert:
            self._db_insert(mean_frame, merged, detid)
        return mean_frame, merged

    # TODO: combine detid, mean_frame, and event_data_dict into a data structure?
    def _db_insert(self, mean_frame = None, event_data_dict = None, detid = None, event_data_getter = None):
        """
        TODO
        """
        #import database
        data_dict =\
            {q.attribute: q.attribute_value for q in self.query}
        data_dict['runs'] = tuple(self.runs)
        if event_data_getter is not None:
            event_data_getter_name = event_data_getter.__name__
            label = self.label + event_data_getter_name
            data_dict['event_data_getter'] = event_data_getter_name
        else:
            label = self.label
        if detid is not None:
            data_dict['detid'] = detid
            data_dict['data'] = (mean_frame, event_data_dict)
            data_dict['event_data_getter'] = database.dumps_b2a(event_data_getter)
        data_dict['label'] = label
        database.mongo_insert_derived_dataset(data_dict)


def query_list(attribute_param_tuples):
    return [construct_query(*tup) for tup in attribute_param_tuples]

def get_derived_datset_labels():
    import database
    return database.mongo_get_all_derived_datasets().keys()

# TODO: This will eventually be moved into an LK20-specific script that
# wraps mecana.py.
materials_ref_match_dict = {
    "Fe3O4": r"Fe.*O(?!.*HEF)",
    "Fe3O4HEF": r"Fe.*O.*HEF",
    "MgO": r"MgO(?!.*HEF)",
    "MgOHEF": r"MgO.*HEF",
    "Diamond": r"Diamond",
    "Graphite": r"Graphite(?!.*HEF)",
    "GraphiteHEF": r"Graphite.*HEF",
    "Dark": r"Dark",
    "Lab6": r"lab.*6"
}

# TODO: move dependence on detid to DataSet.evaluate()
def main(query_list_list = None, detid = 'si', event_filter = None, event_filter_detid = None, ):
    """
    For example, invoke with:
    >>> main([['material', r"Fe3O4"], ['runs', 400, 405]])
    """
    def lk20_material_specifier_substitute(qlist):
        if qlist[1] in materials_ref_match_dict:
            print "Applying string regex substitution: %s --> %s" % (qlist[1], materials_ref_match_dict[qlist[1]])
            qlist[1] = materials_ref_match_dict[qlist[1]]
            
        return tuple(qlist)
    if query_list_list is not None:
        q = query_list(map(lk20_material_specifier_substitute, query_list_list))
    else:
        q = query_list([('material', r".*Fe3O4.*"), ('transmission', 0.05, 0.8)])
    dataset = DataSet(q)
    dataset.evaluate(detid)

#    probe = dataset.evaluate(event_data_getter = eval('config.si_spectrometer_probe'))
#    pump = dataset.evaluate(event_data_getter = eval('config.si_spectrometer_pump'))
#    extract_eventdata = lambda a: map(lambda x: x[2], utils.flatten_dict(a[1]))
#    plt.scatter(extract_eventdata(pump), extract_eventdata(probe))
#    plt.xlabel('pump intensity')
#    plt.ylabel('probe intensity')
#    plt.show()
#    return probe, pump

if __name__ == '__main__':
    main()

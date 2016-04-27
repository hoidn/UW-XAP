"""
For querying datasets by spreadsheet attribute values.
"""

import sys
import re
import config
import numpy as np
import matplotlib.pyplot as plt
import logbook
import utils
import summarymetrics
import data_access
import database
from recordclass import recordclass
from output import rprint
import pdb


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
    d = logbook.get_attribute_dict(logbook_only = True)
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
        bash_special = r"|&;<>()$`\"' \t\n!"
        pat = param1.encode('string-escape')
        pat_bashsafe = filter(lambda c: c not in bash_special, pat)
        label = '-'.join(map(str, [attribute, pat_bashsafe]))
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
        self.event_filter = event_filter
        self.event_filter_detid = event_filter_detid
        if label is None:
            query_strings = [q.label for q in query]
            if event_filter is not None:
                try:
                    filter_identifier = '-'.join(event_filter.params)
                except:
                    filter_identifier = database.hash(utils.random_float())
                try:
                    filter_label =\
                        event_filter.label
                except AttributeError:
                    filter_label =\
                        '-filter-' + event_filter.__name__ + '-' +\
                        filter_identifier + '-' + str(event_filter_detid)
            else:
                filter_label = ''
            self.label = '-'.join(query_strings) + filter_label
        else:
            self.label = label
        # Store this entire data structure
        self._store() 
        # Store a dict mapping self.label to the query parameters
        self._db_insert() 

    def _store(self):
        """
        Store this object to MongoDB.
        """
        rprint( "Storing DataSet. Runs: %s" % str(self.runs))
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
        mean_frame, merged =\
            data_access.get_data_and_filter(self.label, detid,
            event_data_getter = event_data_getter, event_filter = self.event_filter,
            event_filter_detid = self.event_filter_detid)
        if insert:
            self._db_insert(mean_frame, merged, detid)
        return mean_frame, merged

    # TODO: combine detid, mean_frame, and event_data_dict into a data structure?
    def _db_insert(self, mean_frame = None, event_data_dict = None, detid = None, event_data_getter = None):
        """
        TODO
        """
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

    def runfilter(self, filter_func):
        """
        Filter self.runs using filter_func.
        """
        self.runs = filter(filter_func, self.runs)
        self._store()

    # TODO: this method makes the assumption query, event_filter, and event_filter_detid
    # methods are consistent (which requires a match on certain query parameters but not
    # others). Should handle cases where this doesn't hold. This should probably be done
    # by implementing a union method for Query.
    def union(self, other, label):
        """
        Return a Dataset that's the union of self and other and insert it
        into MongoDB.
        """
        if not isinstance(other, DataSet):
            raise ValueError("Argument other must be of type DataSet")
        new_ds = DataSet(self.query, self.event_filter, self.event_filter_detid, immediate_insert = False)
        merged_runs = list(set(self.runs) | set(other.runs))
        new_ds.runs = merged_runs
        new_ds.label = label
        return new_ds


def query_list(attribute_param_tuples):
    return [construct_query(*tup) for tup in attribute_param_tuples]

def get_derived_datset_labels():
    import database
    return database.mongo_get_all_derived_datasets().keys()



def parse_list_of_strings_to_query(slist, partial = ()):
    """
    Given a list of strings, group the strings into tuples, each corresponding to a
    query, and convert string values into numeric ones where possible. The resulting
    tuples may be used to construct Query instances using construct_query.
    """
    def _parse(slist, partial = ()):
        """
        This function destroys the input list
        """
        def parse_query_tuples(tup):
            if not tup:
                return tup
            new = []
            new.append(tup[0])
            for elt in tup[1:]:
                try:
                    new.append(int(elt))
                except ValueError:
                    try:
                        new.append(float(elt))
                    except ValueError:
                        return [tup]
            return [tuple(new)]
        if len(partial) > 3:
            raise ValueError("Invalid query subsequence: %s" % str(partial))
        if not slist:
            return parse_query_tuples(partial)
        next = slist.pop(0)
        if next in logbook.all_logbook_attributes():
            if len(partial) ==  1:
                raise ValueError("Invalid query subsequence: %s" % str(partial))
            if partial:
                return parse_query_tuples(partial) + _parse(slist, partial = (next,))
            else:
                return _parse(slist, partial = (next,))
        else:
            return _parse(slist, partial = partial + (next,))
    return _parse([s for s in slist])

def main(query_string_list, event_filter = None, event_filter_detid = None, label = None):
    # TODO: This will eventually be moved into an LK20-specific script that
    # wraps mecana.py.
    # LK20-specific: allow filtering based range of values of the 'delay' attribute,
    # which is evaluated using xtcav.py
    if 'delay' in query_string_list:
        delay_index = query_string_list.index('delay')
        delay_min, delay_max = query_string_list[delay_index + 1], query_string_list[delay_index + 2]
        query_string_list = query_string_list[:delay_index] + query_string_list[delay_index + 3:]
    if not label:
        label = '_'.join(query_string_list)
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
    substituted_query_string_list =\
        [materials_ref_match_dict[k] if k in materials_ref_match_dict else k
        for k in query_string_list]
    q = query_list(parse_list_of_strings_to_query(substituted_query_string_list))
    dataset = DataSet(q, event_filter = event_filter, event_filter_detid = event_filter_detid, label = label)
    if 'delay_min' in locals():
        import xtcav
        def delay_filter(run_number):
            return float(delay_min) < xtcav.get_delay(run_number) <= float(delay_max)
        dataset.runfilter(delay_filter)
    rprint(dataset.label)
    return dataset

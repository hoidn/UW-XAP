# Author: O. Hoidn

import time
import re
import sys
import json
import webbrowser
import logging
import argparse
import hashlib
import pdb

import httplib2
import os.path


from apiclient import discovery
from oauth2client import client
from oauth2client.file import Storage

import gspread
import pandas as pd
import dill
import numpy as np
from atomicfile import AtomicFile

import utils
import database

import config
from output import rprint

# TODO: logbook doesn't properly sync when labels are changed
# TODO: run range queries fail if the requested range is a subset
# of an existing dataset.

# Format specification for column headers in logbook. Column descriptions:
# runs: range of run numbers in the format 'integer' or 'integer-integer'.
#   Used by: all.
# labels: dataset label, an arbitrary string. This is the only column of which
#   there can be more than one (i.e. a dataset may have more than one label).
#   Used by: all.
# transmission: The fraction of XFEL photons transmitted through upstream filters,
#   in decimal format. Used by: xrd.
# focal_size: focal spot size diameter, in microns. Used by: xrd.
# filter_func: name of a function, defined in config.py, that returns an event-
#   filtering function given 0, 1, or two arguments provided in the param1 and
#   param2 columns. Optional; used by: all.
# param1: first parameter to constructor for event filtering function.
#   Used by: all.
# param2: second parameter to constructor for event filtering function.
#   Used by: all.
# filter_det: detector id of detector whose data is fed to filter_func for
#   event-based filtering. Used by: all
PROPERTY_REGEXES = {'runs': r'.*[rR]un.*', 'transmission': r'.*[tT]ransmission.*|.*[aA]ttenuation.*',
     'focal_size': r'.*[Ss]ize.*', 'labels': r'.*[lL]abel.*|.*[hH]eader.*',
    'param1': r'.*[pP]aram1.*', 'param2': r'.*[pP]aram2.*',
    'param3': r'.*[pP]aram3.*', 'param4': r'.*[pP]aram4.*',
    'filter_det': r'.*[fF]ilter.*[dD]et.*', 'filter_func': r'.*[fF]ilter.*[fF]unc.*',
    'background': r'.*[bB]ackground.*', 'material': r'.*[mM]aterial.*|.*[sS]ample.*'}
HEADERS = [k for k in PROPERTY_REGEXES]

# Format for the flag that designates logbook header row
HEADER_REGEX = r'.*[hH]eader.*'

def get_property_key(col_title):
    for k, v in PROPERTY_REGEXES.iteritems():
        if re.match(v, col_title):
            return k
    raise KeyError(col_title + ": no match found")

# TODO: support the addition of authentication tokens for new users.
# TODO: need a mechanism for indicating bad rows from the logbook itself.

def acquire_oauth2_credentials(secrets_file):
    """
    Flows through OAuth 2.0 authorization process for obtaining Google API
    access credentials for a google account's Drive spreadsheets. This requires
    user authentication.

    Parameters:
        secrets_file : str
        secrets file contains this application's client ID and secret
    """
    flow = client.flow_from_clientsecrets(
        secrets_file,
        scope='https://spreadsheets.google.com/feeds',
        redirect_uri='urn:ietf:wg:oauth:2.0:oob')
    
    auth_uri = flow.step1_get_authorize_url()
    webbrowser.open(auth_uri)
    
    auth_code = raw_input('Enter the authentication code: ')
    
    credentials = flow.step2_exchange(auth_code)
    return credentials

# utility functions
def prefix_add(*args):
    return reduce(lambda x, y: x + y, args)

def list_vstack(*args):
    return reduce(prefix_add, args)

def get_cell_coords(sheet_list2d, regex):
    for i, row in enumerate(sheet_list2d):
        for j, cell in enumerate(row):
            if re.search(regex, cell):
                return i, j
    raise ValueError(regex + ": matching cell not found")

def regex_indexes(regex, col_titles):
    # indices of columns matching the regex
    return [i for i, title in enumerate(col_titles) if re.search(regex, title)]

@utils.memoize(timeout = 10)
def get_logbook_data(url, sheet_number = 0):
    """
    Given a worksheet URL, return a two-tuple consisting of:
        -A list column names from the sheets, with order and values matching
            the keys of PROPERTY_REGEXES.
        -The spreasheet data as a list, in the corresponding order.
    """
    storage =  Storage(utils.resource_path('data/credentials'))
    credentials = storage.get()
    gc = gspread.authorize(credentials)
    document = gc.open_by_url(url)
    worksheets = document.worksheets()
    def process_one_sheet(sheet):
        """
        given a worksheet, return the data as list with order and contents
        matching the keys of property_regexes.

        Returns None if the sheet doesn't contain at least two rows of values.
        """
        raw_data = sheet.get_all_values()
        col_titles, values = spreadsheet_header_body(raw_data)
        if (not col_titles) or (not values):
            return None
        else:
            return col_titles, values
    return filter(lambda x: x, [process_one_sheet(sheet) for sheet in worksheets])


def parse_float(flt_str):
    return float(flt_str)

def parse_focal_size(flt_str):
    # TODO: document the allowable notation 
    if 'best focus' in flt_str:
        return config.best_focus_size
    else:
        if 'um' in flt_str:
            flt_str = flt_str.strip('um')
        try:
            return parse_float(flt_str)
        except ValueError:
            paranthetical_match = re.findall(r".*?\(([0-9]+) *um\).*", flt_str)
            if not paranthetical_match:
                raise ValueError("Incorrect format for focal spot size string")
            else:
                return float(paranthetical_match[0])

def parse_string(string):
    return string

def parse_run(run_string):
    """
    Given a string from the "runs" column in the logbook, of the
    format "abcd" or "abcd-efgh", returns a tuple of ints
    denoting all run numbers in the dataset

    Returns ValueError on an incorrectly-formatted string.
    """
    def parse_consecutive(runrange_string):
        split = runrange_string.split('-')
        try:
            split = map(int, split) # values convertible to ints?
            if len(split) == 1:
                return tuple(range(split[0], split[0] + 1))
            elif len(split) == 2:
                return tuple(range(split[0], split[1] + 1))
            else:
                raise ValueError("Invalid run range format: ", run_string)
        except:
            raise ValueError("Invalid run range format: ", run_string)
    if not run_string:
        return ()
    else:
        run_string = run_string.strip()
        split_ranges = run_string.split(',')
        runrange_tuples =\
            [parse_consecutive(s)
            for s in split_ranges]
        return tuple(set(reduce(lambda x, y: x + y, runrange_tuples)))

parser_dispatch = {'runs': parse_run, 'transmission': parse_float,
    'param1': parse_float, 'param2': parse_float, 'param3': parse_float,
    'param4': parse_float, 'filter_det': parse_string,
    'labels': parse_string, 'focal_size': parse_focal_size, 'filter_func': parse_string,
    'background': parse_string, 'material': parse_string}

def spreadsheet_header_body(sheet_list2d):
    """
    Returns: header, body
    header : 1d list
        The attribute keys corresponding to the contents of the sheet's
        header row, with superfluous columns removed.
    body: 2d list
        Everything in the spreadsheet below the header row, with superfluous
        columns removed.
    """
    if not sheet_list2d:
        return None, None
    try:
        header_i, header_j = get_cell_coords(sheet_list2d, HEADER_REGEX)
    except ValueError:
        header_i = 0
    header = sheet_list2d[header_i]
    # column indices matching one of the header regexes
    valid_columns =\
        reduce(lambda x, y: x + y,
            map(lambda regex: regex_indexes(regex, header),
                PROPERTY_REGEXES.values()))
    row_extract = lambda row: [elt for i, elt in enumerate(row) if i in valid_columns]
    remove_columns = lambda arr2d: [tuple(row_extract(row)) for row in arr2d]
    if len(sheet_list2d) < header_i + 2:
        return row_extract(header), None
    else:
        return row_extract(header), remove_columns(sheet_list2d[header_i + 1:])

def get_label_mapping_one_sheet(col_titles, data):
    label_dict = {}
    enumerated_titles = list(enumerate(col_titles))
    enumerated_labels = filter(lambda pair: pair[1] and re.search(PROPERTY_REGEXES['labels'], pair[1]), enumerated_titles)
    enumerated_properties = filter(lambda pair: pair[1] and not re.search(PROPERTY_REGEXES['labels'], pair[1]), enumerated_titles)

    def insert_one_label(row, label_value):
        local_dict = label_dict.setdefault(label_value, {})
        for k, property in enumerated_properties:
            property_key = get_property_key(property)
            if property_key == 'runs':
                try:
                    val = local_dict.setdefault(property_key, ())
                    local_dict[property_key] = val + parse_run(row[k])
                    # Delete possible duplicates
                    local_dict[property_key] = tuple(set(local_dict[property_key]))
                except ValueError:
                    rprint( "Malformed run range: ", row[k])
            # TODO: make this non-obvious behaviour clear to the user.
            elif property and (property not in local_dict) and row[k]:
                try:
                    local_dict[property_key] = parser_dispatch[property_key](row[k])
                except ValueError, e:
                    rprint( "Malformed attribute: %s" % e)

    for i, row in enumerate(data):
        for j, label in enumerated_labels:
            label_value = row[j]
            if label_value:
                insert_one_label(row, row[j])
            # Also insert an 'anonymous' label for this single row
            insert_one_label(row, database.hash(str(row)))
    return label_dict


# TODO TODO: add a mechanism for cache invalidation when stuff is inserted into
# MongoDB.
@utils.memoize(timeout = 1)
def get_attribute_dict(logbook_only = False):
#    if utils.isroot():
#(        print "Querying MongoDB")
    logbook = database.mongo_get_logbook_dict()
    if logbook_only:
        return logbook
    else:
        derived = database.mongo_get_all_derived_datasets()
        return utils.merge_dicts(logbook, derived)


def get_label_runranges():
    # TODO: continue here
    """
    Return a dictionary mapping each user-supplied label string from
    the google spreadsheet logbook to its corresponding groups of run numbers.

    Output type: Dict mapping strings to lists of tuples.
    """
    complete_dict = get_attribute_dict()
    labels_to_runs = {}
    for label, d in complete_dict.iteritems():
        labels_to_runs[label] = d['runs']
    return labels_to_runs



def get_label_dict(label):
    """
    Return the attribute dict associated with a logbook label.
    """
    if not config.use_logbook:
        raise AttributeError("Logbook not available (disabled in config.py)")
    complete_dict = get_attribute_dict()
    def runs_to_label(run_range):
        """
        Given a run range, look for a label whose run range matches
        and return it. If serveral matches are found, that with the smallest
        number of runs and largest number of attributes (in that order of sorting)
        is returned. If a matching label isn't found, return None.
        """
        red = lambda x, y: x + y
        # TODO: poorly-abstracted...
        filtered_dict = {k: v for k, v in complete_dict.iteritems() if v['runs'] != (None,)}
        labels_to_runtuples = {lab: tuple(get_all_runlist(lab)) for lab in
            filtered_dict}

        # Find the best matching superset label
        best_runset = None
        # TODO: this doesn't work properly if there are colliding labels with
        # same set of runs.
        runtuples_to_labels = {v: k for k, v in labels_to_runtuples.items()}
        target_set = set(run_range)
        for runtuple in runtuples_to_labels:
            if target_set <= set(runtuple):
                if best_runset is None or len(runtuple) < len(best_runset):
                    best_runset = runtuple
        if best_runset:
            return runtuples_to_labels[best_runset]
        else:
            return None

    if label not in complete_dict:
        try:
            runs = parse_run(label)
        except ValueError:
            raise ValueError("label: " + label + " is neither a label nor a correctly-formated run range")
        if runs_to_label(runs) is not None:
            label = runs_to_label(runs)
        else:
            # TODO: stale error message here
            raise KeyError("label: " + label + " is neither a label nor a valid range of run numbers")
    return complete_dict[label]

def all_logbook_attributes():
    d = get_attribute_dict()
    keys = set()
    for attribute_values in d.values():
        for k in attribute_values:
            keys.add(k)
    return list(keys)

@utils.memoize_timeout(timeout = 30)
def get_label_attribute(label, property):
    """
    Return the value of a label's property.
    """
    if property == 'runs':
        try:
            label_runs = parse_run(label)
            return list(label_runs)
        except:
            pass
    label_dict = get_label_dict(label)
    try:
        return label_dict[property]
    except KeyError:
        raise KeyError("attribute: " + property + " of label: " + label + " not found")

def get_run_attribute(run_number, attribute):
    """
    Return the value of an attribute for the given run number.
    Raise a KeyError if it isn't found.

    run_number : int
        A run number.
    """
    return get_label_attribute(str(run_number), attribute)

def eventmask_params(label):
    handles = ['param1', 'param2', 'param3', 'param4']
    result = []
    for p in handles:
        try:
            result.append(get_label_attribute(label, p))
        except KeyError:
            pass
    return result

def get_all_runlist(label):
    """
    Get list of run numbers associated with a label.

    A label may be either a string specified in the google drive logbook or
    a run range of the format 'abcd' or 'abcd-efgh'.
    """
    try:
        runs = parse_run(label)
        return list(runs)
    except: # except what?
        mapping = get_label_runranges()
        # list of tuples denoting run ranges
        # TODO: reorder this and remove fname as a parameter throughout this
        # module once spreadsheet synchronization has been sufficiently tested.
        try:
            groups = mapping[label]
            return list(groups)
        except KeyError:
            # TODO: make sure that the run number exists
            rprint( "logbook label " + label + ": not found")
            try:
                runs = parse_run(label)
            except ValueError:
                raise ValueError(label + ': dataset label not found')
            return list(runs)

def label_mapping_to_datasets(mapping):
    """
    Returns a list of lists of query.DataSet instances. 

    Each sublist corresponds to one key/label in mapping. For sublist contains:
        -One DataSet containing all the runs in the label.
        -One dataset for each run number corresponding to the label.

    As a side effect, all these DataSet instances are inserted into MongoDB. 
    """
    # TODO: modify query.existing_dataset_by_label so that it parses run range
    # specifiers of the form 'a-b' and a,b,c.
    import query
    def make_single_run_dataset(label, run_number):
        """
        Return a function that takes a `mode` parameter (either 'runs' or 'background')
        and returns a dataset.
        """
        def new_dataset(mode = 'runs'):
            import copy
            new = copy.deepcopy(mapping[label])
            new['runs'] = tuple([run_number])
            if mode == 'background':
                new['background'] = ()
            return query.DataSet.from_logbook_label_dict(new, str(run_number))
        return new_dataset

    def make_datasets_one_label(label):
#        if label == 'evaltest':
#            pdb.set_trace()
        run_datasets = [query.DataSet.from_logbook_label_dict(mapping[label], label)] +\
                [make_single_run_dataset(label, run)('runs') for run in mapping[label]['runs']]
        # If background subtraction runs were specified in the logbook these need
        # to be added as well
        if 'background' in mapping[label]:
            try:
                extra_datasets =\
                    [make_single_run_dataset(label, run)('background')
                    for run
                    in parse_run(mapping[label]['background'])]
            except ValueError:
                extra_datasets = []
            return extra_datasets + run_datasets
        else:
            return run_datasets

    return map(make_datasets_one_label, mapping)

def spreadsheet_mapping(url):
    sheet_headers_bodies = get_logbook_data(url)
    mapping_list =\
        [get_label_mapping_one_sheet(col_titles, data)
        for col_titles, data in sheet_headers_bodies]
    return utils.merge_dicts(*mapping_list)


def main(url = config.url):
    
    # TODO: correct the output of spreadsheet_mapping so that this isn't necessary.
    def correct_format(sub_dictionary):
        if 'runs' in sub_dictionary:
            if all(isinstance(r, int) for r in sub_dictionary['runs']):
                return True
        return False

    def filter_mapping(d):
        return {k: v for k, v in d.iteritems() if isinstance(v, dict) and correct_format(v)}

    while True:
        topic = config.expname
        mapping = spreadsheet_mapping(url)
        rprint( mapping)
        database.mongo_insert_logbook_dict(mapping)
        label_mapping_to_datasets(filter_mapping(spreadsheet_mapping(url)))
        time.sleep(3)

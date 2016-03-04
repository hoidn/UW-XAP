# Author: O. Hoidn

import time
import re
import sys
import json
import webbrowser
import logging
import argparse
import hashlib
import ipdb

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
        return parse_float(flt_str)

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
        return (None, None)
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
    invalid_count = 0
    enumerated_titles = list(enumerate(col_titles))
    enumerated_labels = filter(lambda pair: pair[1] and re.search(PROPERTY_REGEXES['labels'], pair[1]), enumerated_titles)
    enumerated_properties = filter(lambda pair: pair[1] and not re.search(PROPERTY_REGEXES['labels'], pair[1]), enumerated_titles)
    for i, row in enumerate(data):
        for j, label in enumerated_labels:
            if label:
                label_value = row[j]
                if not label_value:
                    label_value = database.hash(str(row))
                    invalid_count += 1
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
                            print "Malformed run range: ", row[k]
                    # TODO: make this non-obvious behaviour clear to the user.
                    elif property and (property not in local_dict) and row[k]:
                        try:
                            local_dict[property_key] = parser_dispatch[property_key](row[k])
                        except ValueError, e:
                            print "Malformed attribute: %s" % e
    return label_dict


#@utils.memoize(timeout = 5)
def get_attribute_dict():
#    if utils.isroot():
#        print "Querying MongoDB"
    logbook = database.mongo_get_logbook_dict()
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
        and return it. If a matching label isn't found, return None.
        """
        red = lambda x, y: x + y
        # TODO: poorly-abstracted...
        filtered_dict = {k: v for k, v in complete_dict.iteritems() if v['runs'] != (None,)}
        labels_to_runtuples = {lab: tuple(get_all_runlist(lab)) for lab in
            filtered_dict}
        runtuples_to_labels = {v: k for k, v in labels_to_runtuples.items()}
        target_set = set(run_range)
        for runtuple in runtuples_to_labels:
            if target_set <= set(runtuple):
                return runtuples_to_labels[runtuple]
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
    #ipdb.set_trace()
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
            print "logbook label " + label + ": not found"
            try:
                runs = parse_run(label)
            except ValueError:
                raise ValueError(label + ': dataset label not found')
            return list(runs)

def spreadsheet_mapping(url):
    sheet_headers_bodies = get_logbook_data(url)
    mapping_list =\
        [get_label_mapping_one_sheet(col_titles, data)
        for col_titles, data in sheet_headers_bodies]
    return utils.merge_dicts(*mapping_list)


def main(url = config.url):
    while True:
        topic = config.expname
        mapping = spreadsheet_mapping(url)
        print mapping
        database.mongo_insert_logbook_dict(mapping)
        time.sleep(1)

# Author: O. Hoidn

import numpy as np
import sys
import pandas as pd
import psget
import os
import pdb
import sys
import zmq
import dill
import re
import hashlib

import utils
import logbook
import database
import config
import query
from output import log

from joblib import Memory
memory = Memory(cachedir='cache/data_access/', verbose=0)

# TODO: make logbook data not required for labels that can be parsed as run
# ranges.

XTC_REGEX = r"/reg/d/psdm/" + config.exppath + r"/xtc/" + config.xtc_prefix + "-r([0-9]{4})-s01-c00.xtc"

"""
Module for accessing data associated with logbook-specified run group labels,
using a mapping published by a running instance of logbook.main.
"""


def get_dataset_attribute_map(label):
    try:
        return logbook.get_label_dict(label)
    except ValueError:
        return database.mongo_get_all_derived_datasets()[label]


def get_all_runs(exppath = config.exppath):
    """
    Return numbers of all runs that have been written to the xtc directory.
    """
    all_files_in_xtc = os.listdir('/reg/d/psdm/' + exppath + '/xtc')
    subbed = map(lambda fname: re.sub(os.path.basename(XTC_REGEX), r"\1", fname),\
            all_files_in_xtc)
    result = map(int, [name for name in subbed if len(name) == 4])
    result.sort()
    return result


def get_dataset(dataset_identifier):
    """
    Given a DataSet instance OR a dataset label, return the
    corresponding dataset.
    """
    if isinstance(dataset_identifier, str):
        return query.existing_dataset_by_label(dataset_identifier)
    else:
        return dataset_identifier

#@memory.cache
def eval_dataset(dataset_identifier, detid, event_data_getter = None, event_mask = None,
        dark_frame = None, **kwargs):
    """
    dataset_identifier : string or query.DataSet
        The dataset for which to get data. If this parameter is a string, it
        is interpreted as a dataset label.
    detid : str
        The detector from which to extract data.
    event_data_getter : function
        Function to evaluate over each event.
    event_mask : dict
        Mapping of the format {run numbers -> {event number -> bools}}, indicating which
        events to include and exclude.

    Returns a DataResult instance.
    """
    dataset = get_dataset(dataset_identifier)
    runList = dataset.runs
    if detid in config.nonarea:
        subregion_index = None
    else:
        try:
            subregion_index = config.detinfo_map[detid].subregion_index
        except KeyError, e:
            raise ValueError("Invalid detector id: %s" % detid)
    return psget.get_signal_many_parallel(
        runList, detid, event_data_getter = event_data_getter,
        event_mask = event_mask, subregion_index = subregion_index,
        dark_frame =  dark_frame, **kwargs)
        #print "event data is: ", event_data

def get_dark_dataset(dataset_identifier):
    """
    Return the dataset of either (1) the dark run associated with the
    given dataset based on the logging spreadsheet or (2), if the former isn't
    available, the most proximate preceding dark run.
    """
    dataset = get_dataset(dataset_identifier)
    def autofind_dark():
        darks = query.DataSet.from_query(query.query_list([('material', r".*[dD]ark.*")])).runs
        start_run = np.min(dataset.runs)
        preceding_darks = filter(lambda x: x < start_run, darks)
        if not preceding_darks:
            raise KeyError("No dark frames preceding dataset: %s" % dataset.label)
        else:
            closest_dark = np.max(preceding_darks)
            return str(closest_dark)
    try:
        darklabel = dataset.get_attribute('background')
    except KeyError:
        darklabel = autofind_dark()
    dark_dataset = query.existing_dataset_by_label(darklabel)
    log( "using dark subtraction run: ", darklabel)
    return dark_dataset

#@memory.cache
@utils.eager_persist_to_file('cache/dataccess/epr')
def eval_dataset_and_filter(dataset_identifier, detid, event_data_getter = None,
        darksub = True, frame_processor = None, **kwargs):
    """
    # TODO: update this. Make it clear that this function is the public interface.
    """
    dataset = get_dataset(dataset_identifier)

    def get_darkframe(detid):
        if darksub:
            try:
                dark_dataset = get_dark_dataset(dataset)
                bg_result =  eval_dataset(dark_dataset, detid)
                return bg_result.mean
                #return unsubtracted.bgsubtract(bg_result.mean)
            except KeyError:
                if utils.isroot():
                    log( "No background label found")
                return None
        else:
            return None

    # dataset.event_filter is a function that takes a np array and returns a boolean
    if dataset.event_filter:
        mask_result = eval_dataset(dataset, dataset.event_filter_detid,
                event_data_getter = dataset.event_filter,
                dark_frame = get_darkframe(dataset.event_filter_detid))
        # unpack elements of a DataResult instance
        _, event_mask = mask_result
        sum_true = sum(mask_result.flat_event_data())
        log( "Event mask True entries: ", sum_true, "Total number of events: ", mask_result.nevents())
        return eval_dataset(dataset, detid,
            event_data_getter = event_data_getter, event_mask = event_mask,
            dark_frame = get_darkframe(detid), frame_processor =
            frame_processor)
    else:
        if utils.isroot():
            log( "!!!!!!!!!!!!!!!!!!")
            log( "Dataset %s: No event filter provided." % dataset.label)
            log( "!!!!!!!!!!!!!!!!!!")
        return eval_dataset(dataset, detid,
            event_data_getter = event_data_getter,
            dark_frame = get_darkframe(detid), frame_processor =
            frame_processor)


#def flux_constructor(label):
#    size = get_dataset_attribute_value(label, 'focal_size')
#    return lambda beam_energy: beam_energy * get_dataset_attribute_value(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))

def event_data_dict_to_list(event_data_dict):
    """
    Converts the dict-based representation of event data for a label to
    a flat list of event data objects.
    """
    run_dicts = event_data_dict.values()
    return reduce(lambda x, y: x + y, [d.values() for d in run_dicts])


#def main(label, fname = 'labels.txt'):
#    eval_dataset(label, 1)
#    eval_dataset(label, 2)
#
#if __name__ == '__main__':
#    main(sys.argv[1])

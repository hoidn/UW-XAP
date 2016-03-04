# Author: O. Hoidn

import numpy as np
import sys
import pandas as pd
import avg_bgsubtract_hdf
import os
import pdb
import sys
import zmq
import dill
import re
import ipdb
import hashlib

import utils
import logbook
import database
import config

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

def get_dataset_attribute_value(label, attribute):
    """
    Get the value of an attribute belonging to a dataset label. The dataset
    may be one defined in the logging spreadsheet, or it may be a derived
    dataset.
    """
    try:
        value = logbook.get_label_attribute(label, attribute)
        if isinstance(value, list):
            value = tuple(value)
        return value
    except (KeyError, ValueError, AttributeError), e:
        try:
            return database.get_derived_dataset_attribute(label, attribute)
        except KeyError, e2:
            raise KeyError("logbook.get_label_attribute(): %s\ndatabase.get_dataset_attribute_value(): %s" % (e, e2))

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


@utils.eager_persist_to_file('cache/data_access/get_label_data/')
def get_label_data(label, detid, default_bg = None, override_bg = None,
    event_data_getter = None, event_mask = None, **kwargs):
    """
    Takes a label corresponding to either:
        -A group of runs, or
        -A derived dataset resulting from a previous query.
    Returns:
    signal : np.ndarray
        The detector readout averaged over all events processed.
    event_data : dict
        A list of objects returned by evaluating 
        event_data_getter on each event frame in the dataset.
    #TODO: finish docstring
    """
    runList = logbook.get_all_runlist(label)
    if detid in config.nonarea:
        subregion_index = None
    else:
        subregion_index = config.detinfo_map[detid].subregion_index
    output = avg_bgsubtract_hdf.get_signal_many_parallel(
        runList, detid, event_data_getter = event_data_getter,
        event_mask = event_mask, subregion_index = subregion_index,
        **kwargs)
    signal, event_data = output
    if event_data_getter is None:
        return signal, None
    else:
        return signal, event_data
        #print "event data is: ", event_data

def get_dark_label(label):
    """
    Return the label of either (1) the dark run associated with the
    given label in the logging spreadsheet or (2), if the former isn't
    available, the most proximate preceding dark run.
    """
    def autofind_dark():
        import query
        darks = query.DataSet(query.query_list([('material', r".*[dD]ark.*")])).runs
        start_run = np.min(get_dataset_attribute_value(label, 'runs'))
        preceding_darks = filter(lambda x: x < start_run, darks)
        if not preceding_darks:
            raise KeyError("No dark frames preceding dataset: %s" % label)
        else:
            closest_dark = np.max(preceding_darks)
            return str(closest_dark)
    def get_label_darkframe():
        return get_dataset_attribute_value(label, 'background')
    try:
        darklabel = get_label_darkframe()
    except KeyError:
        darklabel = autofind_dark()
    print "using dark subtraction run: ", darklabel
    return darklabel

def get_data_and_filter(label, detid, event_data_getter = None,
    event_filter = None, event_filter_detid = None):
    """
    # TODO: update this. Make it clear that this function is the public interface.
    """
    def get_event_mask(filterfunc, detid = None):
        """
        TODO
        """
        # filterfunc is a function that takes a np array and returns a boolean
        if detid is None:
            detid = get_dataset_attribute_value(label, 'filter_det')
        imarray, event_data = get_label_data(label, detid,
            event_data_getter = filterfunc)
        return event_data

    try:
        if event_filter:
            event_mask = get_event_mask(event_filter, detid = event_filter_detid)
        else:
            args = logbook.eventmask_params(label)
            try:
                funcstr = get_dataset_attribute_value(label, 'filter_func')
                filterfunc = eval('config.' + funcstr)(*args)
                filter_detid = get_dataset_attribute_value(label, 'filter_det')
                print "DETID IS ", filter_detid
                print "ARGS ARE ", args
                print "FUNCSTR IS", funcstr
            except AttributeError:
                raise ValueError("Function " + funcstr + " not found, and no filter_function/filter_detid in config.py")
            event_mask = get_event_mask(filterfunc, detid = filter_detid)
        merged_mask = utils.merge_dicts(*event_mask.values())
        sum_true = sum(map(lambda d2: sum(d2.values()), event_mask.values()))
        n_events = sum(map(lambda d: len(d.keys()), event_mask.values()))
        print "Event mask True entries: ", sum_true, "Total number of events: ", n_events
        imarray, event_data =  get_label_data(label, detid,
            event_data_getter = event_data_getter, event_mask = event_mask)
    except (KeyError, AttributeError), e:
        if utils.isroot():
            print "!!!!!!!!!!!!!!!!!!"
            print "WARNING: Event filtering will not be performed."
            print e
            print "!!!!!!!!!!!!!!!!!!"
        imarray, event_data =  get_label_data(label, detid,
            event_data_getter = event_data_getter)
    try:
        dark_label = get_dark_label(label)
        bg, _ =  get_label_data(dark_label, detid)
        return imarray - bg, event_data
    except KeyError:
        if utils.isroot():
            print "No background label found"
        return imarray, event_data


def flux_constructor(label):
    size = get_dataset_attribute_value(label, 'focal_size')
    return lambda beam_energy: beam_energy * get_dataset_attribute_value(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))

def event_data_dict_to_list(event_data_dict):
    """
    Converts the dict-based representation of event data for a label to
    a flat list of event data objects.
    """
    run_dicts = event_data_dict.values()
    return reduce(lambda x, y: x + y, [d.values() for d in run_dicts])

# TODO: give this function a home.
#def flux(beam_energy):
#    size = get_dataset_attribute_value(label, 'focal_size')
#    flux = 1e-3 * beam_energy * get_dataset_attribute_value(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))
#    return flux

def main(label, fname = 'labels.txt'):
    get_label_data(label, 1)
    get_label_data(label, 2)

if __name__ == '__main__':
    main(sys.argv[1])

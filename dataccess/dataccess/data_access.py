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

def get_dataset_attribute_value(label, attribute):
    """
    Get the value of an attribute belonging to a dataset label. The dataset
    may be one defined in the logging spreadsheet, or it may be a derived
    dataset.
    """
    try:
        return logbook.get_label_attribute(label, attribute)
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
    try:
        runList = logbook.get_all_runlist(label)
    # look for matching derived dataset and return it if possible
    except ValueError, e:
        from dataccess import database
        dataset = database.mongo_query_derived_dataset(label, detid)
        if dataset is None:
            raise ValueError(str(e) + '\nNo matching derived dataset found.')
        signal, event_data = dataset
        return signal, event_data
        #print "Logbook label not found. Searching derived datasets."

    # Compute logbook-based dataset
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

#@utils.eager_persist_to_file('cache/data_access/get_label_data_and_filter/')
def get_data_and_filter_logbook(label, detid, event_data_getter = None,
    event_filter = None, event_filter_detid = None):
    """
    # TODO: update this. Make it clear that this function is the public interface.
    """
    def get_background():
        """
        Returns background frame.

        Raises KeyError if background label is not found.
        """
        bg_label = get_dataset_attribute_value(label, 'background')
        print "using dark subtraction: ", bg_label
        bg, _ =  get_label_data(bg_label, detid)
        return bg
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
            ipdb.set_trace()
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
    except Exception, e:
        if utils.isroot():
            print "!!!!!!!!!!!!!!!!!!"
            print "WARNING: Event filtering will not be performed."
            print e
            print "!!!!!!!!!!!!!!!!!!"
        imarray, event_data =  get_label_data(label, detid,
            event_data_getter = event_data_getter)
    try:
        bg = get_background()
        return imarray - bg, event_data
    except KeyError:
        if utils.isroot():
            print "No background label found"
        return imarray, event_data

def get_data_and_filter(label, detid, event_data_getter = None,
    event_filter = None, event_filter_detid = None):
    try:
        return get_data_and_filter_logbook(label, detid, event_data_getter = event_data_getter,
            event_filter = event_filter, event_filter_detid = event_filter_detid)
    except:# TODO: catch specific exceptions TODO
        try:
            return database.mongo_query_derived_dataset(label, detid,
                event_data_getter = event_data_getter)
        except KeyError:
            raise ValueError("%s: no matching derived dataset found." % label)

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

def query_event_data(label, detid, flux_min, flux_max, mode = 'all'):
    def flux(beam_energy):
        size = get_dataset_attribute_value(label, 'focal_size')
        flux = 1e-3 * beam_energy * get_dataset_attribute_value(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))
        return flux
    def flux_filter(beam_energy):
        return flux_min < flux(beam_energy) < flux_max
    if mode == 'all':
        imarray, event_data = get_data_and_filter(label, detid, event_filter = flux_filter, event_filter_detid = 'GMD', event_data_getter = lambda x: flux(x))
        return event_data
    elif mode == 'mean':
        imarray, event_data = get_data_and_filter(label, detid, event_filter = flux_filter, event_filter_detid = 'GMD')
        return imarray
    else:
        raise ValueError("Invalid mode")

def main(label, fname = 'labels.txt'):
    get_label_data(label, 1)
    get_label_data(label, 2)

if __name__ == '__main__':
    main(sys.argv[1])

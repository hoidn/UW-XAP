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
import config

# TODO: make logbook data not required for labels that can be parsed as run
# ranges.

XTC_REGEX = r"/reg/d/psdm/" + config.exppath + r"/xtc/" + config.xtc_prefix + "-r([0-9]{4})-s01-c00.xtc"

"""
Module for accessing data associated with logbook-specified run group labels,
using a mapping published by a running instance of logbook.main.
"""

# TODO: description of the defined properties and the modules that use them

def make_labels(fname = 'labels.txt', min_cluster_size = 2):
    """
    Generate list of time-clustered run ranges in a text file. Pairs with 
    get_labels()

    This needs to be run once before invoking the other functions in this module
    """
    clusters = filter(lambda x: len(x) >= min_cluster_size, avg_bgsubtract_hdf.get_run_clusters())
    if os.path.exists(fname):
        raise ValueError("file " + fname + "exists")
    # pairs of start and end run numbers
    bounds = np.array(map(lambda cluster: np.array([cluster[0], cluster[-1]]), clusters))
    np.savetxt(fname, np.ndarray.astype(bounds, int), '%04d', header = 'start run, end run, label1, label2', delimiter = ',')
    return bounds


@utils.memoize(timeout = 5)
def get_pub_logbook_dict():
    # Socket to talk to server
    port = config.port
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    print "Waiting for data on ZMQ pub socket, port ", port
    socket.connect ("tcp://pslogin03:%s" % port)
    topicfilter = config.expname
    socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
    messagedata = socket.recv()[len(config.expname):]
    socket.close()
    return dill.loads(messagedata)

def get_label_runranges():
    # TODO: continue here
    """
    Return a dictionary mapping each user-supplied label string from
    the google spreadsheet logbook to its corresponding groups of run numbers.

    Output type: Dict mapping strings to lists of tuples.
    """
    complete_dict = get_pub_logbook_dict()
    labels_to_runs = {}
    for label, d in complete_dict.iteritems():
        labels_to_runs[label] = d['runs']
    return labels_to_runs

def get_label_property(label, property):
    """
    Return the value of a label's property.
    """
    if property == 'runs':
        try:
            label_runs = logbook.parse_run(label)
            return list(label_runs)
        except:
            pass
    if not config.use_logbook:
        raise AttributeError("Logbook not available (disabled in config.py)")
    complete_dict = get_pub_logbook_dict()
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
            runs = logbook.parse_run(label)
        except ValueError:
            raise ValueError("label: " + label + " is neither a label nor a correctly-formated run range")
        if runs_to_label(runs) is not None:
            label = runs_to_label(runs)
        else:
            raise KeyError("label: " + label + " is neither a label nor a valid range of run numbers")
    label_dict = complete_dict[label]
    try:
        return label_dict[property]
    except KeyError:
        raise KeyError("attribute: " + property + " of label: " + label + " not found")


def eventmask_params(label):
    handles = ['param1', 'param2', 'param3', 'param4']
    result = []
    for p in handles:
        try:
            result.append(get_label_property(label, p))
        except KeyError:
            pass
    return result

def get_all_runlist(label, fname = 'labels.txt'):
    """
    Get list of run numbers associated with a label.

    A label may be either a string specified in the google drive logbook or
    a run range of the format 'abcd' or 'abcd-efgh'.
    """
    try:
        runs = logbook.parse_run(label)
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
            print "label " + label + " not found"
            try:
                runs = logbook.parse_run(label)
            except ValueError:
                raise ValueError(label + ': dataset label not found')
            return list(runs)
        

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

def get_label_data(label, detid, default_bg = None, override_bg = None,
    event_data_getter = None, event_mask = None, **kwargs):
    """
    Given a label corresponding to a group of runs, returns:
        averaged data, event data, 
    where event data is a list of objects returned by evaluating 
    event_data_getter on each event frame in the dataset.
    #TODO: finish docstring
    """
    def concatenated_runlists(lab):
        if lab:
            # convert from numpy type to int after concatenating
            return tuple(map(int, get_all_runlist(lab, fname = fname)))
        else:
            return None # TODO: why?
        
    runList = get_all_runlist(label)
    if not runList:
        raise ValueError(label + ': no runs found for label')
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

def get_data_and_filter(label, detid, event_data_getter = None,
    event_filter = None, event_filter_detid = None):
    """
    # TODO: update this
    """
    def get_background():
        """
        Returns background frame.

        Raises KeyError if background label is not found.
        """
        bg_label = get_label_property(label, 'background')
        bg, _ =  get_label_data(bg_label, detid)
        return bg
    def get_event_mask(filterfunc, detid = None):
        """
        TODO
        """
        # filterfunc is a function that takes a np array and returns a boolean
        if detid is None:
            detid = get_label_property(label, 'filter_det')
        imarray, event_data = get_label_data(label, detid,
            event_data_getter = filterfunc)
        return event_data

    try:
        if event_filter:
            event_mask = get_event_mask(event_filter, detid = event_filter_detid)
        else:
            args = eventmask_params(label)
            try:
                funcstr = get_label_property(label, 'filter_func')
                filterfunc = eval('config.' + funcstr)(*args)
                filter_detid = get_label_property(label, 'filter_det')
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
        print "No background label found"
        return imarray, event_data



def flux_constructor(label):
    size = get_label_property(label, 'focal_size')
    return lambda beam_energy: beam_energy * get_label_property(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))

def event_data_dict_to_list(event_data_dict):
    """
    Converts the dict-based representation of event data for a label to
    a flat list of event data objects.
    """
    run_dicts = event_data_dict.values()
    return reduce(lambda x, y: x + y, [d.values() for d in run_dicts])

def query_event_data(label, detid, flux_min, flux_max, mode = 'all'):
    def flux(beam_energy):
        size = get_label_property(label, 'focal_size')
        flux = 1e-3 * beam_energy * get_label_property(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))
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

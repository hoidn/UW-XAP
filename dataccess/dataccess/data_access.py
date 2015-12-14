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

import utils
import logbook
import config

XTC_REGEX = r"/reg/d/psdm/" + config.exppath + r"/xtc/" + config.xtc_prefix + "-r([0-9]{4})-s01-c00.xtc"

"""
Module for accessing data associated with logbook-specified run group labels,
using a mapping published by a running instance of logbook.main.
"""

# TODO: description of the defined properties and the modules that use them

# For ZMQ TCP communication to get logbook data
PORT = config.port


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
    #bounds = map(lambda cluster: "%s-%s"%(str(cluster[0]), str(cluster[-1])), clusters)
    np.savetxt(fname, np.ndarray.astype(bounds, int), '%04d', header = 'start run, end run, label1, label2', delimiter = ',')
    return bounds

#def get_label_map(fname = 'labels.txt', **kwargs):
#    """
#    Return a dictionary mapping each user-supplied label string from
#    labels.txt to its corresponding groups of run numbers. The 
#    values default to strings based on the run ranges.
#
#    Output type: Dict mapping strings to lists of tuples.
#    """
#    if not os.path.exists(fname):
#        make_labels(fname = fname)
#        print "File",fname,"not found. It will be created."
#    labels = {}
#    labdat = np.array(pd.read_csv(fname, delimiter = ','))
#    #labdat = np.genfromtxt(fname, dtype = None)
#    shape = np.shape(labdat)
#    if len(shape) != 2 or shape[1] > 4:
#        raise StandardError(fname + ' : incorrect format. Must be no more than 4 comma-delimited columns.')
#    for row in labdat:
#        run_range = tuple(map(int, row[:2]))
#        # remove whitespace
#        if isinstance(row[2], str) and row[2].strip() != '' and row[2].strip() != 'None':
#            labels.setdefault(row[2].strip(), []).append(run_range)
#        if isinstance(row[3], str) and row[3].strip() != '' and row[3].strip() != 'None':
#            labels.setdefault(row[3].strip(), []).append(run_range)
#        labels.setdefault("%s-%s"%run_range, []).append(run_range)
#    return labels

@utils.memoize(timeout = 3)
def get_pub_logbook_dict():
    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    print "Waiting for data..."
    socket.connect ("tcp://pslogin03:%s" % PORT)
    topicfilter = ""
    socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
    messagedata = socket.recv()
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
    complete_dict = get_pub_logbook_dict()
    try:
        label_dict = complete_dict[label]
        try:
            return label_dict[property]
        except KeyError:
            raise KeyError("attribute: " + property + " of label: " + label + " not found")
    except KeyError: 
        raise KeyError("label: " + label + " not found")


def get_all_runlist(label, fname = 'labels.txt'):
    """
    Get list of run numbers associated with a label.

    A label may be either a string specified in the google drive logbook or
    a run range of the format 'abcd' or 'abcd-efgh'.
    """

    mapping = get_label_runranges()
    # list of tuples denoting run ranges
    # TODO: reorder this and remove fname as a parameter throughout this
    # module once spreadsheet synchronization has been sufficiently tested.
    try:
        groups = mapping[label]
        return [range(runRange[0], runRange[1] + 1) for runRange in groups]
    except KeyError:
        # TODO: make sure that the run number exists
        print "label " + label + " not found"
        label_range = logbook.parse_run(label)
        return [range(label_range[0], label_range[1] + 1)]
        
        #raise KeyError("label " + label + " not found")

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
    separate = False, event_data_getter = None, event_filter = None, **kwargs):
    """
    Given a label corresponding to a group of runs, returns an array of
    background-subtracted data.
    #TODO: finish docstring
    """
    def concatenated_runlists(lab):
        if lab:
            # convert from numpy type to int after concatenating
            return tuple(map(int, np.concatenate(get_all_runlist(lab, fname = fname))))
        else:
            return None # TODO: why?
        
    #signal, bg = None, None
    default_bg_runlist = concatenated_runlists(default_bg)
    override_bg_runlist = concatenated_runlists(override_bg)
    groups = get_all_runlist(label)
    for runList in groups:
        output = avg_bgsubtract_hdf.get_signal_bg_many_apply_default_bg(
            runList, detid, default_bg = default_bg_runlist, override_bg =
            override_bg_runlist, event_data_getter = event_data_getter,
            event_filter = event_filter, **kwargs)
        newsignal, newbg, event_data = output
        try:
            signal += newsignal
            bg += newbg
        except NameError:
            signal, bg = newsignal.copy(), newbg.copy()
    if separate:
        return (signal) / float(len(groups)), bg / float(len(groups))
    if event_data_getter is None:
        return (signal - bg) / float(len(groups)), None
    else:
        return (signal - bg) / float(len(groups)), event_data

def main(label, fname = 'labels.txt'):
    get_label_data(label, 1)
    get_label_data(label, 2)

if __name__ == '__main__':
    main(sys.argv[1])

# Author: O. Hoidn

import numpy as np
import sys
import pandas as pd
import avg_bgsubtract_hdf
import os
import pdb
import sys

#print sys.path


"""
module for accessing already-processed data by run group labels specified in 
parameters file (default name: labels.txt)
"""

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

def get_label_map(fname = 'labels.txt', **kwargs):
    """
    Return a dictionary mapping each user-supplied label strings from
    labels.txt to its corresponding groups of run numbers. The 
    values default to strings based on the run ranges.

    Output type: Dict mapping strings to lists of tuples.
    """
    if not os.path.exists(fname):
        make_labels(fname = fname)
        print "File",fname,"not found. It will be created."
    labels = {}
    labdat = np.array(pd.read_csv(fname, delimiter = ','))
    #labdat = np.genfromtxt(fname, dtype = None)
    shape = np.shape(labdat)
    if len(shape) != 2 or shape[1] > 4:
        raise StandardError(fname + ' : incorrect format. Must be no more than 4 comma-delimited columns.')
    for row in labdat:
        run_range = tuple(map(int, row[:2]))
        # remove whitespace
        if isinstance(row[2], str) and row[2].strip() != '' and row[2].strip() != 'None':
            labels.setdefault(row[2].strip(), []).append(run_range)
        if isinstance(row[3], str) and row[3].strip() != '' and row[3].strip() != 'None':
            labels.setdefault(row[3].strip(), []).append(run_range)
        labels.setdefault("%s-%s"%run_range, []).append(run_range)
    return labels


def get_all_runlist(label, fname = 'labels.txt'):
    """
    Get list of run numbers associated with a label
    """

    mapping = get_label_map(fname = fname)
    # list of tuples denoting run ranges
    try:
        groups = mapping[label]
    except KeyError:
        raise KeyError("label " + label + " not found")
    return [range(runRange[0], runRange[1] + 1) for runRange in groups]

#def get_label_componenets(label):
#    ldict = get_label_map()
#    runlists = itertools.chain(*np.unique([get_all_runlist(label) for label in ldict.keys()]))
#    for lst in runlists:
    

def get_label_data(label, detid, default_bg = None, override_bg = None, separate = False, fname = 'labels.txt', **kwargs):
    """
    Given a label corresponding to a group of runs, returns an array of
    background-subtracted data
    """
    def concatenated_runlists(lab):
        if lab:
            # convert from numpy type to int after concatenating
            return tuple(map(int, np.concatenate(get_all_runlist(lab, fname = fname))))
        else:
            return None
        
    #signal, bg = None, None
    default_bg_runlist = concatenated_runlists(default_bg)
    override_bg_runlist = concatenated_runlists(override_bg)
    groups = get_all_runlist(label, fname = fname)
    for runList in groups:
        newsignal, newbg = avg_bgsubtract_hdf.get_signal_bg_many_apply_default_bg(runList, detid, default_bg = default_bg_runlist, override_bg = override_bg_runlist, **kwargs)
        try:
            signal += newsignal
            bg += newbg
        except NameError:
            signal, bg = newsignal.copy(), newbg.copy()
    if separate:
        return (signal) / float(len(groups)), bg / float(len(groups))
    return (signal - bg) / float(len(groups))

def main(label, fname = 'labels.txt'):
    get_label_data(label, 1)
    get_label_data(label, 2)

if __name__ == '__main__':
    main(sys.argv[1])

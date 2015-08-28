import pandas as pd
import re
import pdb
import sys
import math
import numpy as np
import Image
import glob
import argparse
import os

import hdfget
from functools import partial

# TODO: find a permanent solution for the exception raised on lines 200-202 of /reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dill/dill.py. For example, handle the exception properly
# TODO: consider making labels more generic, for example by allowing the user
# to generate derived data and then refer to it by label
# Replace the event-code driven method of detecting blank frames with
# something more reliable (i.e. with fewer edge cases)

# NOTE: it is important to do this AFTER importing hdfget (which in turn imports *
# from psana. Otherwise, for some reason, a segfault occurs.
# Necessary for importing dill, which is a dependency of utils
#sys.path.remove('/reg/g/psdm/sw/external/python/2.7.10/x86_64-rhel5-gcc41-opt/lib/python2.7')
sys.path.insert(1, '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
sys.path.insert(1, '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/pathos-0.2a1.dev0-py2.7.egg')

# The version of multiprocessing installed on the psana system is incompatible
# with pathogen. We need to install multiprocessing locally and push its
# to sys.path
sys.path.insert(1, '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/multiprocess-0.70.3-py2.7-linux-x86_64.egg')
import utils
from pathos.multiprocessing import ProcessingPool

EXPNAME = "mecd6714"
XTC_DIR = '/reg/d/psdm/MEC/mecd6714/xtc/'
# Format string to generate the path to a given run's xtc file. Takes an int.
XTC_NAME = "/reg/d/psdm/MEC/mecd6714/xtc/e441-r%04d-s01-c00.xtc"
# XTC filename glob pattern
XTC_GLOB = "/reg/d/psdm/MEC/mecd6714/xtc/e441-*-s01-c00.xtc"
# XTC filename regex pattern
XTC_REGEX = r"/reg/d/psdm/MEC/mecd6714/xtc/e441-r([0-9]{4})-s01-c00.xtc"
DIMENSIONS_DICT = {1: (400, 400), 2: (400, 400), 3: (830, 825)}

# TODO: memoize timestamp lookup
def get_run_clusters(interval = None, max_interval = 50.):
    """
    Returns a list of lists containing clusters of run numbers whose xtc data 
    files were created within max_interval seconds of each other. Testing with
    LD67 data showed 50 seconds to be a good max_interval.

    interval is a tuple of the form [run_low, run_high) indicating a  range of 
    run numbers to limit this operation to
    """
    key = lambda run_num: os.path.getmtime(XTC_NAME%run_num)
    def cluster(data, maxgap, key = lambda x: x):
        '''Arrange data into groups where successive elements
           differ by no more than *maxgap*

           selector is a function that extracts values for comparison from the
           elements of data

            >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
            [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

            >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
            [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

        '''
        data.sort(key = key)
        groups = [[data[0]]]
        for x in data[1:]:
            if abs(key(x) - key(groups[-1][-1])) <= maxgap:
                groups[-1].append(x)
            else:
                groups.append([x])
        return groups

    all_files_in_xtc = os.listdir('/reg/d/psdm/MEC/mecd6714/xtc')
    subbed = map(lambda fname: re.sub(os.path.basename(XTC_REGEX), r"\1", fname),\
            all_files_in_xtc)
    all_runs = map(int, [name for name in subbed if len(name) == 4])
    if interval:
        runs = range(*interval)
        if not set(runs).issubset(set(all_runs)):
            raise ValueError("invalid range of run numbers given")
    else:
        runs = all_runs
    return cluster(runs, max_interval, key = key)


def outliers(eventlist, blanks, sigma_max = 1.0):
    """
    Return the indices of outliers (not including blank frames) in the list of 
    data arrays, along with a list of 'good' (neither outlier nor blank) indices.

    """
    totalcounts = np.array(map(np.sum, eventlist))
    nonblank_counts_enumerated = filter(lambda (indx, counts): indx not in blanks, enumerate(totalcounts))
    nonblank_counts = map(lambda x: x[1], nonblank_counts_enumerated)
    median = np.median(nonblank_counts)
    std = np.std(nonblank_counts)
    print  'signal levels:', totalcounts
    print median, std
    # indices of the events in ascending order of total signal
    outlier_indices = [i  for i, counts in nonblank_counts_enumerated if np.abs(counts - median) > std * sigma_max]
    good_indices = [i for i in range(len(eventlist)) if ((i not in outlier_indices) and (i not in blanks))]
    return outlier_indices, good_indices


@utils.eager_persist_to_file("cache/get_signal_bg_one_run/")
def get_signal_bg_one_run(runNum, detid = 1, sigma_max = 1.0, **kwargs):
    """
    In:
        runNum: run number
        -sigma_max, max deviation from the mean of the total signal
        levels of events we will retain
        -default_bg: list of run numbers from which to exctract blank frames to use
            as background subtraction.
    Returns the averaged signal and background (based on blank frames) for the 
    events in one run

    # TODO: move this part of the doctring somewhere else
    The event code-based method that hdfget uses to identify blank events does
    not work with 60 Hz data. We work around this by using default_bg for
    background subtraction if it is provided (or no subtraction if it is not).
    """
    def spacing_between(arr):
        """
        Given an array (intended to be an array of indices of blank runs), return the
        interval between successive values (assumed to be constant).

        The array must be of length >= 1

        60 Hz datasets should have an interval of 12 between blank indices, 
        whereas 120Hz datasets should have an interval of 24
        """
        diffs = np.diff(arr)[1:]
        return int(np.sum(diffs))/len(diffs)

    def get_bg(eventlist, vetted_blanks):
        if vetted_blanks:
            return reduce(lambda x, y: x + y, eventlist[vetted_blanks])/len(vetted_blanks)
        return np.zeros(np.shape(eventlist[0]))

    nfiles, eventlist, blanks = hdfget.getImg(detid, runNum, EXPNAME)
    if spacing_between(blanks) == 24:
        # throw out the first blank frame, because it does NOT appear to
        # actually be blank in LD67 runs
        vetted_blanks = blanks[1:]
    else:
        vetted_blanks = []
    outlier, good = outliers(eventlist, vetted_blanks, sigma_max = sigma_max)
    bg = get_bg(eventlist, vetted_blanks)
    signal = reduce(lambda x, y: x + y, eventlist[good])/len(good)
    return signal, bg

@utils.persist_to_file("cache/get_signal_bg_many.p")
def get_signal_bg_many(runList, detid, **kwargs):
    """
    Return the averaged signal and background (based on blank frames) over the given runs
    """
    bg = np.zeros(DIMENSIONS_DICT[detid])
    signal = np.zeros(DIMENSIONS_DICT[detid]) 
    for run_number in runList:
        signal_increment, bg_increment = get_signal_bg_one_run(run_number, detid, **kwargs)
        signal += (signal_increment / len(runList))
        bg += (bg_increment / len(runList))
    return signal, bg

@utils.persist_to_file("cache/get_signal_bg_many_parallel.p")
def get_signal_bg_many_parallel(runList, detid, **kwargs):
    """
    Return the averaged signal and background (based on blank frames) over the given runs
    """
    def mapfunc(run_number):
        return get_signal_bg_one_run(run_number, detid, **kwargs)

    MAXNODES = 12
    pool = ProcessingPool(nodes=min(MAXNODES, len(runList)))
    bg = np.zeros(DIMENSIONS_DICT[detid])
    signal = np.zeros(DIMENSIONS_DICT[detid]) 
    run_data = pool.map(mapfunc, runList)
    for signal_increment, bg_increment in run_data:
        signal += (signal_increment / len(runList))
        bg += (bg_increment / len(runList))
    return signal, bg

def get_signal_bg_many_apply_default_bg(runList, detid, default_bg = None,
override_bg = None):
    """
    wraps get_signal_bg_many, additionally allowing a default background 
    subtraction for groups of runs that lack interposed blank frames

    Inputs:
        default_bg: A list of run numbers to use for bg subtraction of 60Hz runs
        override_bg: A list of run numbers to use for bg subtraction of ALL
        runs

    If both default_bg and override_bg are provided, override_bg is used
    """
    signal, bg = get_signal_bg_many_parallel(runList, detid)
    if override_bg:
        discard, bg = get_signal_bg_many(override_bg, detid)
    # if a default background runs are supplied AND bg is all zeros (meaning
    # dummy values were inserted by get_signal_bg_many)
    elif default_bg and not np.any(bg):
        discard, bg = get_signal_bg_many(default_bg, detid)
    return signal, bg

def process_and_save(runList, detid, **kwargs):
    print "processing runs", runList
    signal, bg = get_signal_bg_many(runList, detid, **kwargs)
    boundaries = map(str, [runList[0], runList[-1]])
    os.system('mkdir -p processed/')
    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_bg.dat", bg)   
    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_signal.dat", signal)
    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_bgsubbed.dat", signal - bg)   
    return signal, bg

def process_all_clusters(detid_list, interval = None):
    clusters = get_run_clusters(interval = interval)
    for c in clusters:
        # For LD67 compatibility: clusters of less than 5 runs correspond to 
        # closely spaced optical pump-probe runs, which we don't want to analyze 
        if len(c) >= 5:
            for detid in detid_list:
                process_and_save(c, detid)


def main(runs, detectorList = [1, 2]):
    for det in detectorList:
#        process_all_clusters(det)
        process_and_save(runs, det)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type = int, nargs = '+',  help = 'run numbers to process')

    args = parser.parse_args()

    main(args.runs)

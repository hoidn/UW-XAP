# Author: O. Hoidn

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
import ipdb

from dataccess import psana_get
from dataccess import toscript
from functools import partial
import config # config.py in local directory

# validate contents of config.py
if not config.exppath:
    raise ValueError("config.exppath: must be string of length > 0")
try:
    expname = config.exppath.split('/')[1]
except:
    raise ValueError("config.exppath: incorrect format")

# TODO: ditto... validate values of other parameters in config.py
# do this here or in the top-level script?

# TODO: don't blindly pass **kwargs
# TODO: find a permanent solution for the exception raised on lines 200-202 of /reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dill/dill.py. For example, handle the exception properly
# TODO: consider making labels more generic, for example by allowing the user
# to generate derived data and then refer to it by label
# Replace the event-code driven method of detecting blank frames with
# something more reliable (i.e. with fewer edge cases)

# NOTE: it is important to do this AFTER importing psana_get (which in turn imports *
# from psana. Otherwise, for some reason, a segfault occurs.
# Necessary for importing dill, which is a dependency of utils
#sys.path.remove('/reg/g/psdm/sw/external/python/2.7.10/x86_64-rhel5-gcc41-opt/lib/python2.7')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/pathos-0.2a1.dev0-py2.7.egg')

# The version of multiprocessing installed on the psana system is incompatible
# with pathogen. We need to install multiprocessing locally and push its
# to sys.path
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/multiprocess-0.70.3-py2.7-linux-x86_64.egg')
#sys.path.insert(1, '/reg/neh/home/ohoidn/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/multiprocess-0.70.3-py2.7-linux-x86_64.egg')
from dataccess import utils
from pathos.multiprocessing import ProcessingPool

#XTC_DIR = '/reg/d/psdm/MEC/' + expname + '/xtc/'
XTC_DIR = '/reg/d/psdm/' + config.exppath + '/xtc/'

## TODO: memoize timestamp lookup
#def get_run_clusters(interval = None, exppath = config.exppath, max_interval = 50.):
#    """
#    Returns a list of lists containing clusters of run numbers whose xtc data 
#    files were created within max_interval seconds of each other. Testing with
#    LD67 data showed 50 seconds to be a good max_interval.
#
#    interval is a tuple of the form [run_low, run_high) indicating a  range of 
#    run numbers to limit this operation to
#    """
#    XTC_NAME = "/reg/d/psdm/" + exppath +  r"/xtc/e441-r%04d-s01-c00.xtc"
#    key = lambda run_num: os.path.getmtime(XTC_NAME%run_num)
#    def cluster(data, maxgap, key = lambda x: x):
#        '''Arrange data into groups where successive elements
#           differ by no more than *maxgap*
#
#           selector is a function that extracts values for comparison from the
#           elements of data
#
#            >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
#            [[1, 6, 9], [100, 102, 105, 109], [134, 139]]
#
#            >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
#            [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
#
#        '''
#        data.sort(key = key)
#        groups = [[data[0]]]
#        for x in data[1:]:
#            if abs(key(x) - key(groups[-1][-1])) <= maxgap:
#                groups[-1].append(x)
#            else:
#                groups.append([x])
#        return groups
#
#    all_runs = get_all_runs(exppath = exppath)
#    if interval:
#        runs = range(*interval)
#        if not set(runs).issubset(set(all_runs)):
#            raise ValueError("invalid range of run numbers given")
#    else:
#        runs = all_runs
#    return cluster(runs, max_interval, key = key)


def filter_events(eventlist, blanks, sigma_max = 1.0):
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


#@utils.eager_persist_to_file("cache/get_signal_bg_one_run/", excluded = ['mode'])
#def get_signal_bg_one_run(runNum, detid = 1, sigma_max = 1.0,
#    event_data_getter = None, event_filter = None, mode = 'interactive', **kwargs):
#    """
#    Returns the averaged signal and background (based on blank frames) for the 
#    events in one run
#
#    The event code-based method that psana_get uses to identify blank events does
#    not work with 60 Hz data. We work around this by using default_bg for
#    background subtraction if it is provided (or no subtraction if it is not).
#
#    Parameters
#    ---------
#    runNum : int
#        run number
#    sigma_max : float
#        max deviation from the mean of the total signal
#        levels of events we will retain
#    default_bg : list of ints
#        list of run numbers from which to exctract blank frames to use
#        as background subtraction.
#    event_data_getter : function
#        takes an event data array and returns an element of event_data
#    event_filter : function
#        takes an event data array and returns True or False, indicating
#        whether to exclude the event from signal and bg
#
#    Returns
#    -------
#    signal : 2-d np.ndarray
#        signal averaged over 'good' events
#    bg : 2-d np.ndarray
#        dark frames averaged over 'good' events
#    event_data : list of arbitrary object
#        output from mapping event_data_getter over all event data. If
#        event_data_getter is None, event_data is None
#    """
#    @toscript.makescript(__file__, "bsub -q psanaq -n 1 -o log.out python %s", 'cache/get_signal_bg_one_run/', mode = mode)
#    def get_signal_bg_one_run_inner(runNum, detid = 1, sigma_max = 1.0,
#    event_data_getter = None, event_filter = None, **kwargs):
#        #ipdb.set_trace()
#        def spacing_between(arr):
#            """
#            Given an array (intended to be an array of indices of blank runs), return the
#            interval between successive values (assumed to be constant).
#
#            The array must be of length >= 1
#
#            60 Hz datasets should have an interval of 12 between blank indices, 
#            whereas 120Hz datasets should have an interval of 24
#            """
#            diffs = np.diff(arr)[1:]
#            return int(np.sum(diffs))/len(diffs)
#
#        def get_bg(eventlist, vetted_blanks):
#            if vetted_blanks:
#                return reduce(lambda x, y: x + y, eventlist[vetted_blanks])/len(vetted_blanks)
#            return np.zeros(np.shape(eventlist[0]))
#
#        nfiles, eventlist, blanks = psana_get.getImg(detid, runNum)
#        if spacing_between(blanks) == 24:
#            vetted_blanks = blanks
#        else:
#            vetted_blanks = []
#        outlier, good = filter_events(eventlist, vetted_blanks, sigma_max = sigma_max)
#        bg = get_bg(eventlist, vetted_blanks)
#        signal = reduce(lambda x, y: x + y, eventlist[good])/len(good)
#        if event_data_getter is None:
#            return signal, bg, []
#        else:
#            event_data = map(event_data_getter, eventlist)
#            return signal, bg, event_data
#    #ipdb.set_trace()
#    return get_signal_bg_one_run_inner(runNum, detid = detid, sigma_max = sigma_max,
#        event_data_getter = event_data_getter, event_filter = event_filter, **kwargs)


@utils.eager_persist_to_file("cache/get_signal_bg_one_run/")
def get_signal_bg_one_run(runNum, detid = 1, sigma_max = 1.0,
event_data_getter = None, event_filter = None, **kwargs):
    #ipdb.set_trace()
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

    nfiles, eventlist, blanks = psana_get.getImg(detid, runNum)
    if spacing_between(blanks) == 24:
        vetted_blanks = blanks
    else:
        vetted_blanks = []
    outlier, good = filter_events(eventlist, vetted_blanks, sigma_max = sigma_max)
    bg = get_bg(eventlist, vetted_blanks)
    signal = reduce(lambda x, y: x + y, eventlist[good])/len(good)
    if event_data_getter is None:
        return signal, bg, []
    else:
        event_data = map(event_data_getter, eventlist)
        return signal, bg, event_data

def get_signal_bg_one_run_smd(runNum, detid, sigma_max = 1.0,
        event_data_getter = None, event_filter = None, **kwargs):
    DIVERTED_CODE = 162
    ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
    det = config.detinfo_map[detid]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    darkevents, event_data = [], []
    darksum= np.zeros(config.detinfo_map[detid].dimensions, dtype = np.double)
    signalsum= np.zeros(config.detinfo_map[detid].dimensions, dtype = np.double)
    def is_darkevent(evr):
        for fifoEvent in evr.fifoEvents():
            # In general, this will incorrectly add i = 0 to darkevents
            if fifoEvent.eventCode() == DIVERTED_CODE:
                return True
        return False

    for nevent, evt in enumerate(ds.events):
        if nevent % size == rank:
            evr = evt.get(EvrData.DataV4, Source('DetInfo(NoDetector.0:Evr.0)'))
            isdark = is_darkevent(evr)
            if isdark:
                darkevents.append(i)
                darksum += det.image(evt)
            else:
                incrememnt = det.image(evt)
                signalsum += increment
                if event_data_getter:
                    event_data.append(event_data_getter(increment))
    nevent += 1

    signalsum_final = np.empty_like(signalsum)
    darksum_final = np.empty_like(darksum)
    comm.Reduce(signalsum, signalsum_final)
    #final_eventdata = []
    if event_data_getter:
        comm.gather(event_data)
    comm.reduce(nevent)
    return signalsum_final, darksum_final, event_data

    

@utils.eager_persist_to_file("cache/avg_bgsubtract_hdf/")
def get_signal_bg_many(runList, detid, event_data_getter = None,
    event_filter = None, **kwargs):
    """
    Return the averaged signal and background, and accumulated event data,
    from running get_signal_bg_one_run over all runs specified.

    Parameters
    ----------
    runList : list of ints
    All others: see get_signal_bg_one_run

    Returns
    ----------
    signal : 2-d np.ndarray
        Averaged signal, subject to filtering by event_filter, and default
        removal of outliers.
    bg : 2-d np.ndarray
        Averaged background.
    event_filter : list of lists
        Accumulated event data.
    """
    bg = np.zeros(config.detinfo_map[detid].dimensions)
    signal = np.zeros(config.detinfo_map[detid].dimensions) 
    event_data = []
    for run_number in runList:
        output_one_run = get_signal_bg_one_run(run_number, detid,
            event_data_getter = event_data_getter, event_filter = event_filter,
            **kwargs)
        signal_increment, bg_increment, event_data_entry = output_one_run
        signal += (signal_increment / len(runList))
        bg += (bg_increment / len(runList))
        if event_data_getter is not None:
            event_data.append(event_data_entry)
    return signal, bg, event_data

@utils.eager_persist_to_file("cache/get_signal_bg_many_parallel/")
def get_signal_bg_many_parallel(runList, detid, event_data_getter = None,
    event_filter = None, **kwargs):
    """
    Parallel version of get_signal_bg_many
    """
    def mapfunc(run_number):
        return get_signal_bg_one_run(run_number, detid, event_data_getter =
            event_data_getter, event_filter = event_filter, **kwargs)

    MAXNODES = 14
    pool = ProcessingPool(nodes=min(MAXNODES, len(runList)))
    try:
        bg = np.zeros(config.detinfo_map[detid].dimensions)
    except KeyError:
        raise KeyError(detid + ': detid not found in config.py')
    signal = np.zeros(config.detinfo_map[detid].dimensions) 
    #run_data = map(mapfunc, runList)
    run_data = pool.map(mapfunc, runList)
    event_data = []
    for signal_increment, bg_increment, event_data_entry in run_data:
        signal += (signal_increment / len(runList))
        bg += (bg_increment / len(runList))
        event_data.append(event_data_entry)
    return signal, bg, event_data

# TODO: fix this mess with event_data_getter
def get_signal_bg_many_apply_default_bg(runList, detid, default_bg = None,
override_bg = None, event_data_getter = None, event_filter = None):
    """
    Wraps get_signal_bg_many, additionally allowing a default background 
    subtraction for groups of runs that lack interposed blank frames

    Inputs:
        default_bg: A list of run numbers to use for bg subtraction of 60Hz runs
        override_bg: A list of run numbers to use for bg subtraction of ALL
        runs

    If both default_bg and override_bg are provided, override_bg is used
    """
    signal, bg, event_data = get_signal_bg_many_parallel(runList, detid,
        event_data_getter = event_data_getter, event_filter = event_filter)
    if override_bg:
        discard, bg, event_data = get_signal_bg_many(override_bg, detid,
            event_data_getter = event_data_getter, event_filter = event_filter)
    # if a default background runs are supplied AND bg is all zeros (meaning
    # dummy values were inserted by get_signal_bg_many)
    elif default_bg and not np.any(bg):
        discard, bg, event_data = get_signal_bg_many(default_bg, detid,
            event_data_getter = event_data_getter, event_filter = event_filter)
    return signal, bg, event_data

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

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('runs', type = int, nargs = '+',  help = 'run numbers to process')
#
#    args = parser.parse_args()
#
#    main(args.runs)

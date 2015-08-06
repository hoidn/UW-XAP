import gc
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

#TODO: find a permanent solution for the exception raised on lines 200-202 of /reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dill/dill.py. For example, handle the exception properly

# NOTE: it is important to do this AFTER importing hdfget (which in turn imports *
# from psana. Otherwise, for some reason, a segfault occurs.
sys.path.insert(1, '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
import utils

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
    return the indices of outliers (not including blank frames) in the list of 
    data arrays, along with a list of 'good' (neither outlier nor blank) indices
    """
    totalcounts = map(np.sum, eventlist)
    nonblank_counts_enumerated = filter(lambda (indx, counts): indx not in blanks, enumerate(totalcounts))
    nonblank_counts = map(lambda x: x[1], nonblank_counts_enumerated)
    mean = np.mean(nonblank_counts)
    std = np.std(nonblank_counts)
    print  'signal levels:', totalcounts
    print mean, std
    # indices of the events in ascending order of total signal
    outlier_indices = [i  for i, counts in nonblank_counts_enumerated if np.abs(counts - mean) > std * sigma_max]
    good_indices = [i for i in range(len(eventlist)) if ((i not in outlier_indices) and (i not in blanks))]
    return outlier_indices, good_indices

def blank_outlier_good(runNum, sigma_max = 1.0, intensity_det_id = 3):
    """
    In:
        run number
        -sigma_max, max deviation from the mean of the total signal
        levels of events we will retain
        -intensity_det_id, the id of the detector we're using to determine outliers
        (defaults to 3, the quad CSPAD)
    Returns indices of blank frames, outliers, and good (neither outlier nor blank)
    frames
    """
    nfiles, eventlist, blanks = hdfget.getImg(intensity_det_id, runNum, EXPNAME)
    outlier, good = outliers(eventlist, blanks, sigma_max = sigma_max)
    return blanks, outlier, good

def get_signal_bg_one_run(runNum, detid, **kwargs):
    """
    Returns the averaged signal and background (based on blank frames) for the 
    events in one run
    """
    blank, outlier, good = blank_outlier_good(runNum, **kwargs)
    nfiles, eventlist, blanks = hdfget.getImg(detid, runNum, EXPNAME)
    bg = reduce(lambda x, y: x + y, eventlist[blank])/len(blank)
    signal = reduce(lambda x, y: x + y, eventlist[good])/len(good)
    return signal, bg

@utils.persist_to_file("cache/get_signal_bg_many.p")
def get_signal_bg_many(runList, detid, **kwargs):
    """
    return the averaged signal and background (based on blank frames) over the given runs
    """
    bg = np.zeros(DIMENSIONS_DICT[detid])
    signal = np.zeros(DIMENSIONS_DICT[detid]) 
    for run_number in runList:
        gc.collect() # collect garbage
        signal_increment, bg_increment = get_signal_bg_one_run(run_number, detid, **kwargs)
        signal += signal_increment
        bg += bg_increment
    return signal, bg

def process_and_save(runList, detid, **kwargs):
    signal, bg = get_signal_bg_many(runList, detid, **kwargs)
    boundaries = map(str, [runList[0], runList[-1]])
    os.system('mkdir -p processed/')
    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_bg.dat", bg)   
    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_signal.dat", signal)
    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_bgsubbed.dat", signal - bg)   
    return signal, bg

def process_all_clusters(detid, interval = None):
    clusters = get_run_clusters(interval = interval)
    for c in clusters:
        # For LD67 compatibility: clusters of less than 5 runs correspond to 
        # closely spaced optical pump-probe runs, which we don't want to analyze 
        # here. TODO: not very pretty
        if len(c) >= 5:
            print "processing runs", c
            process_and_save(c, detid)

def main(detectorList = [1, 2]):
    for det in detectorList:
        process_all_clusters(det)

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('run', type = int, nargs = '+',  help = 'start and end numbers of ranges of runs to process')

    #args = parser.parse_args()
    #if len(args.run)%2 != 0:
#        raise ValueError("number of args must be positive and even (i.e., must specify one or more ranges of runs)")
#    fullRange = []
#    for i in range(len(args.run)/2):
#        fullRange += range(args.run[2 * i], 1 + args.run[2 * i + 1])
#    main(fullRange)
    main()


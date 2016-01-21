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
import random
import ipdb
import dill
import sys
from time import time

random.seed(os.getpid())

import config
if not config.smd:
    from dataccess import psana_get
from psana import *
from PSCalib.GeometryAccess import img_from_pixel_arrays
from Detector.GlobalUtils import print_ndarr
from dataccess import toscript
from functools import partial
import config # config.py in local directory

from mpi4py import MPI
comm = MPI.COMM_WORLD

# validate contents of config.py
if not config.exppath:
    raise ValueError("config.exppath: must be string of length > 0")
try:
    expname = config.exppath.split('/')[1]
except:
    raise ValueError("config.exppath: incorrect format")

# TODO: ditto... validate values of other parameters in config.py
# do this here or in the top-level script?

# Replace the event-code driven method of detecting blank frames with
# something more reliable (i.e. with fewer edge cases)

from dataccess import utils
from pathos.multiprocessing import ProcessingPool

#XTC_DIR = '/reg/d/psdm/MEC/' + expname + '/xtc/'
XTC_DIR = '/reg/d/psdm/' + config.exppath + '/xtc/'


@utils.eager_persist_to_file("cache/get_signal_bg_one_run/")
def get_signal_bg_one_run(runNum, detid = 1, sigma_max = 1000.0,
event_data_getter = None, event_mask = None, **kwargs):
    # TODO: remove sigma_max discrimination
    if detid in config.nonarea:
        return get_signal_bg_one_run_nonarea(runNum, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)

    def filter_events(eventlist, blanks):
        """
        Return the indices of outliers (not including blank frames) in the list of 
        data arrays, along with a list of 'good' (neither outlier nor blank) indices.

        """
        def event_valid(nevent):
            if event_mask:
                run_mask = event_mask[runNum]
                return run_mask[nevent]
            else:
                return True
        totalcounts = np.array(map(np.sum, eventlist))
        nonblank_counts_enumerated = filter(lambda (indx, counts): indx not in blanks, enumerate(totalcounts))
        nonblank_counts = map(lambda x: x[1], nonblank_counts_enumerated)
        median = np.median(nonblank_counts)
        std = np.std(nonblank_counts)
        print  'signal levels:', totalcounts
        print median, std
        # indices of the events in ascending order of total signal
        outlier_indices =\
            [i
            for i, counts in nonblank_counts_enumerated
            if (np.abs(counts - median) > std * sigma_max) or not event_valid(i)]
        good_indices =\
            [i
            for i in range(len(eventlist))
            if ((i not in outlier_indices) and (i not in blanks))]
        return outlier_indices, good_indices

    def spacing_between(arr):
        """
        Given an array (intended to be an array of indices of blank runs), return the
        interval between successive values (assumed to be constant).

        The array must be of length >= 1

        60 Hz datasets should have an interval of 12 between blank indices, 
        whereas 120Hz datasets should have an interval of 24

        To be called ONLY if config.smd == False
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
    outlier, good = filter_events(eventlist, vetted_blanks)
    bg = get_bg(eventlist, vetted_blanks)
    signal = reduce(lambda x, y: x + y, eventlist[good])/len(good)
    if event_data_getter is None:
        return signal, bg, []
    else:
        if event_mask is not None:
            event_data = map(event_data_getter, np.array(eventlist)[np.array(event_mask[runNum], dtype = bool)])
        else:
            event_data = map(event_data_getter, eventlist)
        return signal, bg, event_data


def get_area_detector_subregion(quad, det, evt, detid):
    """
    Extracts data from an individual quad detector.
    """
    if quad>3 : quad = 3
    if quad >= 0:
        if 'Cspad' not in config.detinfo_map[detid].device_name:
            raise ValueError("Can't take subregion of non-CSPAD detector")
        rnum = evt.run()
        geo = det.geometry(rnum)        # for >ana-0.17.5

        # get pixel index array for quad, shape=(8, 185, 388)
        iX, iY = geo.get_pixel_coord_indexes('QUAD:V1', quad)
        print_ndarr(iX, 'iX')
        print_ndarr(iY, 'iY')

        t0_sec = time()

        nda = det.raw(evt)

        print 'Consumed time = %7.3f sec' % (time()-t0_sec)
        #print_ndarr(nda, 'raw')

        # get intensity array for quad, shape=(8, 185, 388)
        nda.shape = (4, 8, 185, 388)
        ndaq = nda[quad,:]
        #print_ndarr(ndaq, 'nda[%d,:]'%quad)

        # reconstruct image for quad
        img = img_from_pixel_arrays(iX, iY, W=ndaq)
        new = np.empty_like(img)
        new[:] = img
        return new
    else:
        if 'Cspad' in config.detinfo_map[detid].device_name:
            return det.image(evt)
        else:
            return det.raw(evt)

# TODO: have this run only on the root rank
def get_signal_bg_one_run_nonarea(runNum, detid,
        event_data_getter = None, event_mask = None, **kwargs):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if config.smd:
        ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
    else:
        ds = DataSource('exp=%s:run=%d:stream=0,1'% (config.expname,runNum))
    def event_valid(nevent):
        if event_mask:
            run_mask = event_mask[runNum]
            return run_mask[nevent]
        else:
            return True
    def mapfunc(event):
        try:
            result = event.get(Lusi.IpmFexV1, Source(config.nonarea[detid])).channel()[0]
        # Event is None, or something like that
        except AttributeError:
            result = np.nan
        return result
    det_values = []
    for nevent, evt in enumerate(ds.events()):
        if  (nevent % size == rank) and event_valid(nevent):
            k = evt.get(Lusi.IpmFexV1, Source(config.nonarea[detid]))
            if k:
                det_values.append(k.channel()[0])
#    det_values =\
#        [evt.get(Lusi.IpmFexV1, Source(config.nonarea[detid])).channel()[0]
#        for nevent, evt in enumerate(ds.events())
#        if event_valid(nevent) and (nevent % size == rank)]
    det_values = reduce(lambda x, y: x + y, comm.allgather(det_values))
#    det_values = utils.mpimap(mapfunc, ds.events())
#    det_values = filter(lambda x: x != np.isnan, det_values)
    event_mean = np.sum(det_values) / len(det_values)
    if event_data_getter:
        event_data = map(event_data_getter, det_values)
    else:
        event_data = []
    return event_mean, 0., event_data

#def get_signal_bg_one_run_nonarea_mproc(runNum, detid,
#        event_data_getter = None, event_mask = None, **kwargs):
#    rank = comm.Get_rank()
#    size = comm.Get_size()
#    if config.smd:
#        ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
#    else:
#        ds = DataSource('exp=%s:run=%d:stream=0,1'% (config.expname,runNum))
#    def event_valid(nevent):
#        if event_mask:
#            run_mask = event_mask[runNum]
#            return run_mask[nevent]
#        else:
#            return True
#    def mapfunc(event):
#        try:
#            result = event.get(Lusi.IpmFexV1, Source(config.nonarea[detid])).channel()[0]
#        # Event is None, or something like that
#        except AttributeError:
#            result = np.nan
#        return result
#    pool = ProcessingPool(nodes=14)
#    det_values = pool.map(mapfunc, ds.events())
#    det_values = filter(lambda x: x != np.isnan, det_values)
#    event_mean = np.sum(det_values) / len(det_values)
#    if event_data_getter:
#        event_data = map(event_data_getter, det_values)
#    else:
#        event_data = []
#    return event_mean, 0., event_data




# TODO: more testing and refactor all of this!
@utils.eager_persist_to_file("cache/get_signal_bg_one_run_smd/")
def get_signal_bg_one_run_smd_area(runNum, detid, subregion_index = -1,
        event_data_getter = None, event_mask = None, **kwargs):
    def event_valid(nevent):
        if event_mask:
            run_mask = event_mask[runNum]
            return run_mask[nevent]
        else:
            return True
    #DIVERTED_CODE = 162
    ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
    det = Detector(config.detinfo_map[detid].device_name, ds.env())
    rank = comm.Get_rank()
    print "rank is", rank
    size = comm.Get_size()
    darkevents, event_data = [], []
    events_processed = 0
    def is_darkevent(evr):
        # temporary patch, valid for LK20 because end station A isn't running
#        for fifoEvent in evr.fifoEvents():
#            # In general, this will incorrectly add i = 0 to darkevents
#            if fifoEvent.eventCode() == DIVERTED_CODE:
#                return True
        return False
    for nevent, evt in enumerate(ds.events()):
        # TODO: testing only, remove later.
#        if nevent > 200:
#            break
        if (nevent % size == rank) and event_valid(nevent):
#            if random.randint(0, 10) != 5:
#                continue

            evr = evt.get(EvrData.DataV4, Source('DetInfo(NoDetector.0:Evr.0)'))
            isdark = is_darkevent(evr)
            #ipdb.set_trace()
            increment = get_area_detector_subregion(subregion_index, det, evt, detid)
            if increment is not None:
                if isdark:
                    darkevents.append(nevent)
                    try:
                        darksum += increment
                    except UnboundLocalError:
                        darksum = increment
                else:
                    try:
                        signalsum += increment
                    except UnboundLocalError:
                        signalsum = np.zeros_like(increment)
                        signalsum += increment
                    if event_data_getter:
                        event_data.append(event_data_getter(increment))
                    events_processed += 1
                    print 'processed event: ', nevent
    try:
        signalsum /= events_processed
    except UnboundLocalError:
        raise ValueError("No events found for det: " + str(detid) + ", run: " + str(runNum))
    signalsum_final = np.empty_like(signalsum)
    comm.Allreduce(signalsum, signalsum_final)
    # TODO: refactor
    darksum_final = np.empty_like(signalsum)
    try:
        darksum /= events_processed
        comm.Allreduce(darksum, darksum_final)
    except UnboundLocalError:
        darksum_final = np.empty_like(signalsum_final)
    darkevents = comm.allgather(darkevents)
    events_processed = comm.allreduce(events_processed)
    if event_data_getter:
        event_data = comm.allgather(event_data)
    print "rank is: ", rank
    if rank == 0:
        print "processed ", events_processed, "events"
        print "dark events: ", darkevents
    if darkevents:
        darkevents = reduce(lambda x, y: x + y, darkevents)
    if event_data:
        event_data = reduce(lambda x, y: x + y, event_data)
    return signalsum_final, darksum_final, event_data


@utils.eager_persist_to_file("cache/get_signal_bg_one_run_smd/")
def get_signal_bg_one_run_smd(runNum, detid, subregion_index = -1,
        event_data_getter = None, event_mask = None, **kwargs):
    if detid in config.nonarea:
        return get_signal_bg_one_run_nonarea(runNum, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
    else: # Assume detid to be an area detector
        return get_signal_bg_one_run_smd_area(runNum, detid, subregion_index = subregion_index,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)



@utils.eager_persist_to_file("cache/get_signal_bg_many_parallel/")
def get_signal_bg_many_parallel(runList, detid, event_data_getter = None,
    event_mask = None, **kwargs):
    """
    Parallel version of get_signal_bg_many
    """
    def mapfunc(run_number):
        return get_signal_bg_one_run(run_number, detid, event_data_getter =
            event_data_getter, event_mask = event_mask, **kwargs)
    def mapfunc_smd(run_number):
        return get_signal_bg_one_run_smd(run_number, detid, event_data_getter =
            event_data_getter, event_mask = event_mask, **kwargs)

    if config.smd:
        run_data = map(mapfunc_smd, runList)
    else:
        MAXNODES = 14
        pool = ProcessingPool(nodes=min(MAXNODES, len(runList)))
        run_data = pool.map(mapfunc, runList)
        #run_data = map(mapfunc, runList)
    event_data = {}
    runindx = 0
    for signal_increment, bg_increment, event_data_entry in run_data:
        try:
            signal += (signal_increment / len(runList))
        except UnboundLocalError:
            signal = signal_increment / len(runList)
        try:
            bg += (bg_increment / len(runList))
        except UnboundLocalError:
            bg = bg_increment / len(runList)
        event_data[runList[runindx]] = event_data_entry
        runindx += 1
        #event_data.append(event_data_entry)
    return signal, bg, event_data

def get_signal_bg_many_apply_default_bg(runList, detid, default_bg = None,
override_bg = None, event_data_getter = None, event_mask = None, **kwargs):
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
        event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
    if override_bg:
        discard, bg, event_data = get_signal_bg_many_parallel(override_bg, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
    # if a default background runs are supplied AND bg is all zeros (meaning
    # dummy values were inserted by get_signal_bg_many)
    elif default_bg and not np.any(bg):
        discard, bg, event_data = get_signal_bg_many_parallel(default_bg, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
    return signal, bg, event_data

#def process_and_save(runList, detid, **kwargs):
#    print "processing runs", runList
#    signal, bg = get_signal_bg_many(runList, detid, **kwargs)
#    boundaries = map(str, [runList[0], runList[-1]])
#    os.system('mkdir -p processed/')
#    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_bg.dat", bg)   
#    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_signal.dat", signal)
#    np.savetxt("./processed/" + "-".join(boundaries) + "_" + str(detid) + "_bgsubbed.dat", signal - bg)   
#    return signal, bg


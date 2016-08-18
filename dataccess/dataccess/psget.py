# Author: O. Hoidn
import pandas as pd
import re
import pdb
import sys
import math
import numpy as np
from PIL import Image
import glob
import argparse
import os
import random
import dill
import sys
from time import time
from collections import namedtuple

import config
from output import log


"""
Module for accessing data from the psana API.
"""

DataResultBase = namedtuple("DataResultBase", "mean, event_data")
class DataResult(DataResultBase):
    """
    Basic datatype representing extracted data for a set of events.

    Attributes: 
    mean : np.ndarray
        Mean of detector readout over a number of events
    event_data : dict
        Maps {run number: {event number: event data}}
    """
    def __new__(cls, mean, event_data_dict):
        self = super(DataResult, cls).__new__(cls, mean, event_data_dict)
        return self

    def flat_event_data(self):
        """
        Return a 1d np.ndarray of event data.
        """
        def extract_event_data(arr2d):
            return arr2d[:, -1]
        return extract_event_data(utils.flatten_dict(self.event_data))

    def nevents(self):
        """ Return the number of events"""
        return np.sum(map(len, self.event_data.values()))

    def bgsubtract(self, bgarr):
        """Return a new instance with bgarr subtracted from self.mean"""
        from copy import deepcopy
        return DataResult(
            self.mean - bgarr,
            deepcopy(self.event_data))

    def __add__(self, other):
        return DataResult(
            self.mean + other.mean,
            utils.merge_dicts(self.event_data, other.event_data))

    def matching_flat_event_data(self, other):
        """
        other : DataResult instance
        Return data of the same type and format as flat_event_data, but exclude
        events that are not contained in other.
        """
        pruned_event_data = utils.prune_dict(self.event_data, other.event_data)
        return DataResult(None, pruned_event_data).flat_event_data()
        

def idxgen(ds):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank==0: log( 'idx mode')
    size = comm.Get_size()
    run = ds.runs().next()
    times = run.times()
    mylength = len(times)//size
    startevt = rank*mylength
    mytimes= times[startevt:(rank+1)*mylength]
    for nevent,t in enumerate(mytimes, startevt):
        yield nevent,run.event(t)

def smdgen(ds):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0: log( 'smd mode')
    for nevent,evt in enumerate(ds.events()):
        if nevent%size == rank: yield nevent,evt

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

XTC_DIR = '/reg/d/psdm/' + config.exppath + '/xtc/'


@utils.eager_persist_to_file("cache/get_signal_one_run/")
def get_signal_one_run(runNum, detid = 1, sigma_max = 1000.0,
event_data_getter = None, event_mask = None, **kwargs):
    # TODO: remove sigma_max discrimination
    if detid in config.nonarea:
        return get_signal_one_run_nonarea(runNum, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)

    def filter_events(eventlist, blanks):
        """
        Return the indices of outliers (not including blank frames) in the list of 
        data arrays, along with a list of 'good' (neither outlier nor blank) indices.

        """
        def event_valid(nevent):
            if event_mask:
                run_mask = event_mask[runNum]
                if nevent in run_mask:
                    return run_mask[nevent]
                else:
                    return False
            else:
                return True
        totalcounts = np.array(map(np.sum, eventlist))
        nonblank_counts_enumerated = filter(lambda (indx, counts): indx not in blanks, enumerate(totalcounts))
        nonblank_counts = map(lambda x: x[1], nonblank_counts_enumerated)
        median = np.median(nonblank_counts)
        std = np.std(nonblank_counts)
        log(  'signal levels:', totalcounts)
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

    nfiles, eventlist, blanks = psana_get.getImg(detid, runNum)
    if spacing_between(blanks) == 24:
        vetted_blanks = blanks
    else:
        vetted_blanks = []
    outlier, good = filter_events(eventlist, vetted_blanks)
    signal = reduce(lambda x, y: x + y, eventlist[good])/len(good)
    if event_data_getter is None:
        return signal, {}
    else:
        if event_mask is not None:
            event_data = map(event_data_getter, np.array(eventlist)[np.array(event_mask[runNum], dtype = bool)])
        else:
            event_data = map(event_data_getter, eventlist)
        return signal, event_data


def get_area_detector_subregion(quad, det, evt, detid):
    """
    Extracts data from an individual quad detector.

    if chip_level_correction, the 50th percentile value for each
    chip is subtracted.
    """
    if quad>3 : quad = 3
    if quad >= 0:
        if 'Cspad' not in config.detinfo_map[detid].device_name:
            raise ValueError("Can't take subregion of non-CSPAD detector")
        rnum = evt.run()
        geo = det.geometry(rnum)        # for >ana-0.17.5

        # get pixel index array for quad, shape=(8, 185, 388)
        iX, iY = geo.get_pixel_coord_indexes('QUAD:V1', quad)
#        print_ndarr(iX, 'iX')
#        print_ndarr(iY, 'iY')

        t0_sec = time()

        nda = det.raw(evt)
        ped = det.pedestals(evt)

        #print 'Consumed time = %7.3f sec' % (time()-t0_sec)
        #print_ndarr(nda, 'raw')

        # get intensity array for quad, shape=(8, 185, 388)
        nda.shape = (4, 8, 185, 388)
        ndaq = nda[quad,:]
        
        ped.shape = (4, 8, 185, 388)
        pedq = ped[quad,:]
# TODO: should we keep chip-level correction disabled?
        try:
            chip_correction = config.chip_level_correction
        except AttributeError, e:
            raise utils.ConfigAttributeError(str(e))
        if chip_correction:
            for chip_pedestal, chip_nda in zip(pedq, ndaq):
                offset = np.percentile(chip_nda, 45) - np.mean(chip_pedestal)
                chip_pedestal += offset
        #print_ndarr(ndaq, 'nda[%d,:]'%quad)

        # reconstruct image for quad
        img = img_from_pixel_arrays(iX, iY, W=ndaq)
        bg = img_from_pixel_arrays(iX, iY, W=pedq)
        new = np.empty_like(img)
        new[:] = (img - bg)
        return new
    else:
        if 'Cspad' in config.detinfo_map[detid].device_name:
            return det.image(evt)
        else:
            return det.raw(evt)

@utils.eager_persist_to_file('cache/avg_bgsubtract_hdf/get_event_data_nonarea')
def get_event_data_nonarea(runNum, detid, **kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if config.smd:
        ds = DataSource('exp=%s:run=%d:idx' % (config.expname, runNum))
        #ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
        #ds = DataSource('exp=%s:run=%d:smd:dir=/reg/d/ffb/%s/xtc:live' % (config.expname, runNum, config.exppath))
    else:
        ds = DataSource('exp=%s:run=%d:stream=0,1'% (config.expname,runNum))
    log( '')
    log( "PROCESSING RUN: ", runNum)
    log( '')
    evtgen = idxgen(ds)
    #evtgen = smdgen(ds)
    det_values = []
    def eval_lusi(evt):
        """LUSI detector reading"""
        k = evt.get(Lusi.IpmFexV1, Source(config.nonarea[detid].src))
        if k:
            det_values.append(k.channel()[0])
    def eval_bld(evt):
        """gas detector average reading"""
        k = evt.get(Bld.BldDataFEEGasDetEnergyV1, Source(config.nonarea[detid].src))
        if k:
            det_values.append(np.mean([k.f_11_ENRC(), k.f_12_ENRC(), k.f_21_ENRC(), k.f_22_ENRC()]))
            #print "appending: ", str([k.f_11_ENRC(), k.f_12_ENRC(), k.f_21_ENRC(), k.f_22_ENRC()])
    for nevent, evt in evtgen:
        if config.testing and nevent % 10 != 0:
            continue
        if config.nonarea[detid].type == 'Lusi.IpmFexV1':
            eval_lusi(evt)
        elif config.nonarea[detid].type == 'Bld.BldDataFEEGasDetEnergyV1':
            eval_bld(evt)
        else:
            raise ValueError("Not a valid non-area detector")
    return reduce(lambda x, y: x + y, comm.allgather(det_values))

#@utils.eager_persist_to_file('cache/avg_bgsubtract_hdf/get_signal_one_run_nonarea')
def get_signal_one_run_nonarea(runNum, detid,
        event_data_getter = None, event_mask = None, **kwargs):
    def event_valid(nevent):
        if event_mask:
            run_mask = event_mask[runNum]
            return run_mask[nevent]
        else:
            return True

    det_values = get_event_data_nonarea(runNum, detid, **kwargs)
    det_values_filtered =\
        [dat
        for i, dat in enumerate(det_values)
        if event_valid(i)]
    event_mean = np.sum(det_values_filtered) / len(det_values_filtered)

    if event_data_getter:
        event_data_list =\
            [event_data_getter(dv, run = runNum)
            for dv in det_values]
        # filtered event data
        event_data =\
            {i: dat
            for i, dat in enumerate(event_data_list)
            if event_valid(i)}
    else:
        event_data = {}
    return event_mean, event_data, len(det_values_filtered)


# TODO: more testing and refactor all of this!
@utils.eager_persist_to_file("cache/get_signal_one_run_smd_area/")
def get_signal_one_run_smd_area(runNum, detid, subregion_index = -1,
        event_data_getter = None, event_mask = None, **kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    def event_valid(nevent):
        if event_mask:
            run_mask = event_mask[runNum]
            if nevent in run_mask:
                return run_mask[nevent]
            else:
                return False
        else:
            return True
    #DIVERTED_CODE = 162
    ds = DataSource('exp=%s:run=%d:idx' % (config.expname, runNum))
    #ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
    #evtgen = smdgen(ds)
    evtgen = idxgen(ds)
    #ds = DataSource('exp=%s:run=%d:smd:dir=/reg/d/ffb/%s/xtc:live' % (config.expname, runNum, config.exppath))
    det = Detector(config.detinfo_map[detid].device_name, ds.env())
    rank = comm.Get_rank()
    log( "rank is", rank)
    size = comm.Get_size()
    event_data = {}
    events_processed = 0
    last = time()
    last_nevent = 0
    for nevent, evt in evtgen:
        if config.testing and nevent % 10 != 0:
            continue
        if event_valid(nevent):
            evr = evt.get(EvrData.DataV4, Source('DetInfo(NoDetector.0:Evr.0)'))
            try:
                increment = get_area_detector_subregion(subregion_index, det, evt, detid, **kwargs)
            except AttributeError:
                continue
            if increment is not None:
                # TODO: modify the non-smd version of this function so that mutation
                # of increment by event_data_getter carries through in the same way
                # (or better yet, refactor so that this current code is reused).
                if event_data_getter:
                    event_data[nevent] = event_data_getter(increment, run = runNum,
                        nevent = nevent)
                try:
                    signalsum += increment
                except UnboundLocalError:
                    signalsum = np.zeros_like(increment).astype('float')
                    signalsum += increment
                events_processed += 1
            if nevent % 100 == 0:
                now = time()
                deltat = now - last
                deltan = nevent - last_nevent
                log( 'processed event: ', nevent, (deltan/deltat) * size, "rank is: ", rank, "size is: ", size)
                last = now
                last_nevent = nevent
    try:
        signalsum_final = np.empty_like(signalsum)
        comm.Allreduce(signalsum, signalsum_final)
        events_processed = comm.allreduce(events_processed)
        signalsum_final /= events_processed
        if event_data_getter:
            event_data = comm.allgather(event_data)
        log( "rank is: ", rank)
        if rank == 0:
            log( "processed ", events_processed, "events")
        if event_data:
            #print event_data
            #event_data = reduce(lambda x, y: x + y, event_data)
            log( 'before merge')
            log( event_data)
            event_data = utils.merge_dicts(*event_data)
        return signalsum_final, event_data, events_processed
    except UnboundLocalError:
        raise ValueError("No events found for det: " + str(detid) + ", run: " + str(runNum) + ": " + str(events_processed))


#@utils.eager_persist_to_file("cache/get_signal_one_run_smd/")
def get_signal_one_run_smd(runNum, detid, subregion_index = -1,
        event_data_getter = None, event_mask = None, **kwargs):
    if detid in config.nonarea:
        return get_signal_one_run_nonarea(runNum, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
    else: # Assume detid to be an area detector
        return get_signal_one_run_smd_area(runNum, detid, subregion_index = subregion_index,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)



@utils.eager_persist_to_file("cache/get_signal_many_parallel/")
def get_signal_many_parallel(runList, detid, event_data_getter = None,
    event_mask = None, **kwargs):
    """
    Parallel version of get_signal_many
    """
    def mapfunc(run_number):
        return get_signal_one_run(run_number, detid, event_data_getter =
            event_data_getter, event_mask = event_mask, **kwargs)
    def mapfunc_smd(run_number):
        return get_signal_one_run_smd(run_number, detid, event_data_getter =
            event_data_getter, event_mask = event_mask, **kwargs)

    if config.smd:
        # Iterate through runs. Bad runs are excluded from the returned
        # data, unless all runs are bad, in which case a ValueError is
        # raised.
        run_data = []
        exceptions = []
        for run in runList:
            try:
                run_data.append(mapfunc_smd(run))
            except ValueError, e:
                exceptions.append(e)
                log( "WARNING: ", e)
        if not run_data:
            if not exceptions:
                msg = 'No runs found'
            else:
                msg = '. '.join(map(str, exceptions))
            raise ValueError(msg)
        elif exceptions:
            log( "WARNING: INVALID RUNS WILL BE EXCLUDED")
    else:
        MAXNODES = 14
        pool = ProcessingPool(nodes=min(MAXNODES, len(runList)))
        run_data = pool.map(mapfunc, runList)
        #run_data = map(mapfunc, runList)
    event_data = {}
    runindx = 0
    total_events = np.sum(map(lambda x: x[2], run_data))
    for signal_increment, event_data_entry, events_processed in run_data:
        try:
            signal += signal_increment * (float(events_processed) / total_events)
        except UnboundLocalError:
            signal = signal_increment * (float(events_processed) / total_events)
        event_data[runList[runindx]] = event_data_entry
        runindx += 1
    return DataResult(signal, event_data)
    #return signal, event_data



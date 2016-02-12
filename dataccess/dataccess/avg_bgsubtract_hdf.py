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

from mpi4py import MPI
comm = MPI.COMM_WORLD

def idxgen(ds):
    rank = comm.Get_rank()
    if rank==0: print 'idx mode'
    size = comm.Get_size()
    run = ds.runs().next()
    times = run.times()
    mylength = len(times)//size
    startevt = rank*mylength
    mytimes= times[startevt:(rank+1)*mylength]
    for nevent,t in enumerate(mytimes, startevt):
        yield nevent,run.event(t)

def smdgen(ds):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0: print 'smd mode'
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

#XTC_DIR = '/reg/d/psdm/MEC/' + expname + '/xtc/'
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
    rank = comm.Get_rank()
    size = comm.Get_size()
    if config.smd:
        ds = DataSource('exp=%s:run=%d:idx' % (config.expname, runNum))
        #ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
        #ds = DataSource('exp=%s:run=%d:smd:dir=/reg/d/ffb/%s/xtc:live' % (config.expname, runNum, config.exppath))
    else:
        ds = DataSource('exp=%s:run=%d:stream=0,1'% (config.expname,runNum))
    print ''
    print "PROCESSING RUN: ", runNum
    print ''
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
    return event_mean, event_data


# TODO: more testing and refactor all of this!
@utils.eager_persist_to_file("cache/get_signal_one_run_smd_area/")
def get_signal_one_run_smd_area(runNum, detid, subregion_index = -1,
        event_data_getter = None, event_mask = None, **kwargs):
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
    print "rank is", rank
    size = comm.Get_size()
    event_data = {}
    events_processed = 0
    last = time()
    last_nevent = 0
    for nevent, evt in evtgen:
        if event_valid(nevent):
            evr = evt.get(EvrData.DataV4, Source('DetInfo(NoDetector.0:Evr.0)'))
            try:
                increment = get_area_detector_subregion(subregion_index, det, evt, detid)
            except AttributeError:
                continue
            if increment is not None:
                try:
                    signalsum += increment
                except UnboundLocalError:
                    signalsum = np.zeros_like(increment)
                    signalsum += increment
                if event_data_getter:
                    #event_data.append(event_data_getter(increment))
                    event_data[nevent] = event_data_getter(increment, run = runNum, nevent = nevent)
                events_processed += 1
            if nevent % 100 == 0:
                now = time()
                deltat = now - last
                deltan = nevent - last_nevent
                print 'processed event: ', nevent, (deltan/deltat) * size, "rank is: ", rank, "size is: ", size
                last = now
                last_nevent = nevent
    try:
        signalsum_final = np.empty_like(signalsum)
        comm.Allreduce(signalsum, signalsum_final)
        events_processed = comm.allreduce(events_processed)
        signalsum_final /= events_processed
        if event_data_getter:
            event_data = comm.allgather(event_data)
        print "rank is: ", rank
        if rank == 0:
            print "processed ", events_processed, "events"
        if event_data:
            #print event_data
            #event_data = reduce(lambda x, y: x + y, event_data)
            print 'before merge'
            print event_data
            event_data = utils.merge_dicts(*event_data)
        return signalsum_final, event_data
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
        run_data = map(mapfunc_smd, runList)
    else:
        MAXNODES = 14
        pool = ProcessingPool(nodes=min(MAXNODES, len(runList)))
        run_data = pool.map(mapfunc, runList)
        #run_data = map(mapfunc, runList)
    event_data = {}
    runindx = 0
    for signal_increment, event_data_entry in run_data:
        try:
            signal += (signal_increment / len(runList))
        except UnboundLocalError:
            signal = signal_increment / len(runList)
        event_data[runList[runindx]] = event_data_entry
        runindx += 1
    return signal, event_data



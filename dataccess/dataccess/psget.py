# Author: O. Hoidn
import pdb
import numpy as np
import os
import random
import time
from collections import namedtuple
import batchjobs
import inspect

import config
import logging
from output import log

frame = inspect.currentframe()

"""
Module for accessing data from the psana API.
"""

POOL_NODES = 12
def get_pool():
    from pathos.multiprocessing import ProcessingPool
    return ProcessingPool(nodes=POOL_NODES)

#def get_pool():
#    from pathos.parallel import ParallelPool as Pool
#    pool = Pool()
#    pool.ncpus = 12
#    pool.servers = ('psanagpu101:4321',)#'psanagpu105:4321',)
#    return pool


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
        # TODO: currently does not work if each event datum is a numpy array rather than
        # a numeric type
        def extract_event_data(arr2d):
            try:
                data_col = arr2d[:, -1]
            except TypeError: # arr2d is a list, not an array
                data_col = np.array([elt[-1] for elt in arr2d])
            if type(data_col[0]) == np.ndarray:
                return np.vstack(data_col)
            return data_col
        return extract_event_data(utils.flatten_dict(self.event_data))

    # TODO docstring
    def iter_event_value_pairs(self):
        event_data_dict = self.event_data
        for run in event_data_dict:
            for nevent in event_data_dict[run]:
                yield event_data_dict[run][nevent]

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

    def intersection(self, other):
        """
        other : DataResult instance
        Return data of the same type and format as flat_event_data, but exclude
        events that are not contained in other.
        """
        pruned_event_data = utils.prune_dict(self.event_data, other.event_data)
        return DataResult(None, pruned_event_data)
        

def idxgen(ds):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank==0: log( 'idx mode')
    size = comm.Get_size()
    run = ds.runs().next()
    times = run.times()
    log('size: '+ str(size))
    mylength = len(times)//size
    startevt = rank*mylength
    mytimes= times[startevt:(rank+1)*mylength]
    for nevent,t in enumerate(mytimes, startevt):
        yield nevent,run.event(t)

#def mproc_process(nevent, run):
#    times = run.times()
#    return run.event(times[nevent])

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
import psana
from PSCalib.GeometryAccess import img_from_pixel_arrays
#from psana.Detector.GlobalUtils import print_ndarr
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

XTC_DIR = '/reg/d/psdm/' + config.exppath + '/xtc/'


@utils.eager_persist_to_file('cache/psget/gsor')
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

        t0_sec = time.time()

        nda = det.raw(evt)
        if nda is None:
            msg = "get_area_detector_subregion: det.raw() returned None"
            log (msg)
            raise AttributeError(msg)
        ped = det.pedestals(evt)
        # documentation: https://confluence.slac.stanford.edu/display/
        # PSDM/Common+mode+correction+algorithms
        cm = det.common_mode_correction(evt, nda - ped, [5, 50])

        #print 'Consumed time = %7.3f sec' % (time()-t0_sec)
        #print_ndarr(nda, 'raw')

        # get intensity array for quad, shape=(8, 185, 388)
        nda.shape = (4, 8, 185, 388)
        ndaq = nda[quad,:]

        cm.shape = nda.shape
        cmq = cm[quad,:]
        
        ped.shape = (4, 8, 185, 388)
        pedq = ped[quad,:]
# TODO: should we keep chip-level correction disabled?
        try:
            chip_correction = config.chip_level_correction
        except AttributeError, e:
            log (str(e))
            raise
               
#        except AttributeError, e:
#            raise utils.ConfigAttributeError(str(e))
        if chip_correction:
            for chip_pedestal, chip_nda in zip(pedq, ndaq):
                offset = np.percentile(chip_nda, 45) - np.mean(chip_pedestal)
                chip_pedestal += offset
        #print_ndarr(ndaq, 'nda[%d,:]'%quad)

        # reconstruct image for quad
        img = img_from_pixel_arrays(iX, iY, W=ndaq)
        bg = img_from_pixel_arrays(iX, iY, W=pedq)
        common = img_from_pixel_arrays(iX, iY, W=cmq)

        new = np.empty_like(img)
        new[:] = (img - (bg - common))
        return new
    else:
        if 'Cspad' in config.detinfo_map[detid].device_name:
            increment = det.image(evt)
        else:
            increment = det.raw(evt)
        if increment is not None:
            return increment.astype('float')
        else:
            return increment

def get_ds(runNum):
    if config.smd:
        return psana.DataSource('exp=%s:run=%d:idx' % (config.expname, runNum))
        #ds = DataSource('exp=%s:run=%d:smd' % (config.expname, runNum))
        #ds = DataSource('exp=%s:run=%d:smd:dir=/reg/d/ffb/%s/xtc:live' % (config.expname, runNum, config.exppath))
    else:
        return psana.DataSource('exp=%s:run=%d:stream=0,1'% (config.expname,runNum))

#@utils.eager_persist_to_file('cache/psget/gedn')
def get_event_data_nonarea(runNum, detid = None, **kwargs):
    log( '')
    log( "PROCESSING RUN: ", runNum)
    log( '')
    if config.multiprocess:
        log('using multiprocessing in get_event_data_nonarea')
        pool = get_pool()
        if config.testing:
            size = 1
        else:
            size = pool.ncpus
        def mapfunc(i):
            ds = get_ds(runNum)
            if i ==0: log( 'idx mode')
            run = ds.runs().next()
            times = run.times()
            log('size: '+ str(size))
            length = len(times)//size
            startevt = i *length
            mytimes= times[startevt:(i +1)*length]
            det_values = []
            for t in mytimes:
                evt = run.event(t)
                det_values = accumulator_nonarea(evt, detid, det_values = det_values)
            return det_values

        gathered = pool.map(mapfunc, range(size))
    else:
        log('using MPI in get_event_data_nonarea')
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        #evtgen = smdgen(ds)
        ds = get_ds(runNum)
        evtgen = idxgen(ds)
        det_values = []
        for nevent, evt in evtgen:
            log('processing' + str(nevent))
            if config.testing and nevent % 10 != 0:
                continue
            det_values = accumulator_nonarea(evt, detid, det_values = det_values)
        log('going to gather')
        gathered = comm.allgather(det_values)
        log('gathered')
    return reduce(lambda x, y: x + y, gathered)

#@utils.eager_persist_to_file('cache/psget/gedn')
def accumulator_nonarea(evt, detid, det_values = []):
    def eval_lusi(evt):
        """LUSI detector reading"""
        k = evt.get(psana.Lusi.IpmFexV1, psana.Source(config.nonarea[detid].src))
        if k:
            return k.channel()[0]
    def eval_bld(evt):
        """gas detector average reading"""
        k = evt.get(psana.Bld.BldDataFEEGasDetEnergyV1, psana.Source(config.nonarea[detid].src))
        if k:
            return np.mean([k.f_11_ENRC(), k.f_12_ENRC(), k.f_21_ENRC(), k.f_22_ENRC()])
    if config.nonarea[detid].type == 'Lusi.IpmFexV1':
        new = eval_lusi(evt)
    elif config.nonarea[detid].type == 'Bld.BldDataFEEGasDetEnergyV1':
        new = eval_bld(evt)
    else:
        raise ValueError("Not a valid non-area detector")
    if new is not None:
        det_values.append(new)
    return det_values

#@memory.cache
def get_signal_one_run_nonarea(runNum, detid = None,
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
def eval_frame_processor(evt, ds, frame_processor, **kwargs):
    def detid_array(detid):
        det = psana.Detector(config.detinfo_map[detid].device_name, ds.env())
        try:
            subregion_index = config.detinfo_map[detid].subregion_index
        except KeyError, e:
            raise ValueError("Invalid detector id: %s" % detid)
        return get_area_detector_subregion(subregion_index, det, evt,
            detid)
    if 'detids' in frame_processor.__dict__:
        log ("detids: %s" % str(frame_processor.detids))
        det_data_dict = {detid: detid_array(detid) for detid in frame_processor.detids}
        det_data_dict.update(kwargs)
        return frame_processor(**det_data_dict)
    else:
        try:
            return frame_processor(detid_array(kwargs['detid']), **kwargs)
        except KeyError, e:
            raise ValueError("kwarg 'detid' must be provided if frame_processor lacks the attribute detids")

def accumulator_area(ds, evt,  nevent, runNum, det, signalsum = None, detid = None, event_data = {}, events_processed = 0,
        dark_frame = None, event_mask = None, frame_processor = None, event_data_getter = None, **kwargs):
    def event_valid(nevent):
        if config.testing and nevent % 10 != 0:
            return False
        if event_mask is not None:
            run_mask = event_mask[runNum]
            if nevent in run_mask:
                return run_mask[nevent]
            else:
                return False
        else:
            return True
    if event_valid(nevent):
        try:
            subregion_index = config.detinfo_map[detid].subregion_index
            increment = get_area_detector_subregion(subregion_index, det, evt,
                detid)
            if dark_frame is not None:
                increment -= dark_frame#.astype('uint16')
            if frame_processor is not None:
                if dark_frame is None:
                    log( 'dark frame provided but will not be applied' )
                #np.save('increment%d.npy' % nevent, increment)
                if 'detid' not in kwargs:
                    kwargs['detid'] = detid
                increment = eval_frame_processor(evt, ds, frame_processor, **kwargs)
                log( "processing frame, event %d" % nevent)
        except (AttributeError, TypeError) as e:
            logging.exception(e)
            if config.testing:
                raise
            increment = None
            #evr = evt.get(EvrData.DataV4, psana.Source('DetInfo(NoDetector.0:Evr.0)'))
        if increment is not None:
            # TODO: modify the non-smd version of this function so that mutation
            # of increment by event_data_getter carries through in the same way
            # (or better yet, refactor so that this current code is reused).
            if event_data_getter:
                event_data[nevent] = event_data_getter(increment, run = runNum,
                    nevent = nevent)
            if signalsum is None:
                signalsum = np.zeros_like(increment).astype('float')
            signalsum += increment
            events_processed += 1
        else:
            if event_valid(nevent):
                log('bad event: %d' % nevent)
    return signalsum, event_data, events_processed

#@utils.eager_persist_to_file('cache/psget/gsorsa')
def get_signal_one_run_smd_area(runNum, detid = None, event_data_getter = None, event_mask = None,
        frame_processor = None, dark_frame = None, **kwargs):
    def multiprocess_func():
        """
        Returns signalsum_final, event_data, events_processed, or None if no
        valid data is found

        signalsum_final : numpy array
            The sum of event data (either raw or transformed by
            frame_processor, if that function is non-None
        event_data : list of dicts
            list of dectionaries generated by evaluating event_data_getter on
            each event datum. One list element per MPI rank.
        events_processed : int
            number of events processed
        """
        log('using multiprocessing in get_signal_one_run_smd_area')
        pool = get_pool()
        if config.testing:
            size = 1
        else:
            size = pool.ncpus
        def mapfunc(i):
            ds = get_ds(runNum)
            det = psana.Detector(config.detinfo_map[detid].device_name, ds.env())
            run = ds.runs().next()
            times = run.times()
            log('size: '+ str(size))
            length = len(times)//size
            startevt = i *length
            mytimes= times[startevt:(i +1)*length]
            for nevent, t in enumerate(mytimes, start = startevt):
                evt = run.event(t)
                try:
                    signalsum, event_data, events_processed = accumulator_area(ds, evt,  nevent, runNum, det,
                            detid = detid, signalsum = signalsum, event_data = event_data,
                            events_processed = events_processed, dark_frame = dark_frame,
                            event_mask = event_mask, frame_processor = frame_processor, event_data_getter = event_data_getter, **kwargs)
                except NameError:
                    signalsum, event_data, events_processed = accumulator_area(ds, evt, nevent, runNum, det, detid = detid, 
                            dark_frame = dark_frame, event_mask = event_mask, frame_processor = frame_processor, event_data_getter = event_data_getter, **kwargs)
            return signalsum, event_data, events_processed 

        if config.testing:
            gathered = map(mapfunc, range(size))
        else:
            try:
                gathered = pool.map(mapfunc, range(size))
            except AssertionError, e: # Can't do nestied multiprocessing with daemonic processes
                log(str(e))
                gathered = map(mapfunc, range(1))
        signalsum_list, event_data_list, events_processed_list  = zip(*gathered)
        signalsum = reduce(lambda x, y: x + y, signalsum_list)
        events_processed = reduce(lambda x, y: x + y, events_processed_list)
        return signalsum, event_data_list, events_processed
    def mpi_func():
        """
        Returns signalsum_final, event_data, events_processed, or None if no
        valid data is found

        signalsum_final : numpy array
            The sum of event data (either raw or transformed by
            frame_processor, if that function is non-None
        event_data : list of dicts
            list of dectionaries generated by evaluating event_data_getter on
            each event datum. One list element per MPI rank.
        events_processed : int
            number of events processed
        """
        log('using MPI in get_signal_one_run_smd_area')
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        #DIVERTED_CODE = 162
        ds = get_ds(runNum)
        det = psana.Detector(config.detinfo_map[detid].device_name, ds.env())
        #evtgen = smdgen(ds)
        evtgen = idxgen(ds)
        #det = Detector(config.detinfo_map[detid].device_name, ds.env())
        rank = comm.Get_rank()
        log( "rank is", rank)
        size = comm.Get_size()
        last = time.time()
        last_nevent = 0
        for nevent, evt in evtgen:
            if config.testing and nevent % 10 != 0:
                continue
            try:
                signalsum, event_data, events_processed = accumulator_area(ds, evt, nevent, runNum,
                        det, signalsum = signalsum, detid = detid,
                        event_data = event_data, events_processed = events_processed, dark_frame = dark_frame,
                        event_mask = event_mask,  frame_processor = frame_processor, event_data_getter = event_data_getter, **kwargs)
            except NameError:
                signalsum, event_data, events_processed = accumulator_area(ds, evt, nevent, runNum, det, detid = detid,
                        dark_frame = dark_frame, event_mask = event_mask, frame_processor = frame_processor, event_data_getter = event_data_getter, **kwargs)
            if nevent % 100 == 0:
                now = time.time()
                deltat = now - last
                deltan = nevent - last_nevent
                log( 'processed event: ', nevent, (deltan/deltat) * size, "rank is: ", rank, "size is: ", size)
                last = now
                last_nevent = nevent
        if signalsum is not None:
            signalsum_final = np.empty_like(signalsum)
            comm.Allreduce(signalsum, signalsum_final)
            events_processed = comm.allreduce(events_processed)
            if event_data_getter:
                event_data = comm.allgather(event_data)
            log( "rank is: ", rank)
            if rank == 0:
                log( "processed ", events_processed, "events")
            return signalsum_final, event_data, events_processed

    if config.multiprocess:
        result = multiprocess_func()
    else:
        result = mpi_func()

    if result is not None:
        signalsum_final, event_data, events_processed = result
        signalsum_final /= events_processed
        if event_data:
            #print event_data
            #event_data = reduce(lambda x, y: x + y, event_data)
            log( 'before merge')
            #log( event_data)
            event_data = utils.merge_dicts(*event_data)
        return signalsum_final, event_data, events_processed
    else:
        raise ValueError("No events found for det: " + str(detid) + ", run: " + str(runNum) + ": " + str(events_processed))


def check_autompi():
    return config.autompi

#@memory.cache
@utils.conditional_decorator(batchjobs.JobPool, check_autompi)
def get_signal_one_run_smd(runNum, detid = None, event_data_getter = None, event_mask = None,
        **kwargs):
    if detid in config.nonarea:
        return get_signal_one_run_nonarea(runNum, detid,
            event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
    else: # Assume detid to be an area detector
        log('calling get_signal_one_run_smd_area')
        result = get_signal_one_run_smd_area(runNum, detid, event_data_getter = event_data_getter, event_mask = event_mask, **kwargs)
        log('called get_signal_one_run_smd_area')
        return result



#@memory.cache
@utils.eager_persist_to_file('cache/psget/gsmp')
def get_signal_many_parallel(runList, detid = None, event_data_getter = None,
    event_mask = None, **kwargs):
    """
    Parallel version of get_signal_many
    """
    def mapfunc(run_number):
        return get_signal_one_run(run_number, detid, event_data_getter =
            event_data_getter, event_mask = event_mask, **kwargs)
    # TODO move the .get()u up one level on the stack
    def mapfunc_smd(run_number):
        log('calling get_signal_one_run_smd')
        return get_signal_one_run_smd(run_number, detid, event_data_getter =
            event_data_getter, event_mask = event_mask, **kwargs)

    if config.smd:
        # Iterate through runs. Bad runs are excluded from the returned
        # data, unless all runs are bad, in which case a ValueError is
        # raised.
        run_data = []
        exceptions = []
        #async_results = [mapfunc_smd(run) for run in runList]
        #for result in async_results:
        for run in runList:
            #try:
            #run_data.append(result)
            log('applying mapfunc_smd')
            if config.autompi:
                run_data.append(mapfunc_smd(run).get())
            else:
                run_data.append(mapfunc_smd(run))
#            except ValueError, e:
#                exceptions.append(e)
#                log( "WARNING: ", e)
        if not run_data:
            if not exceptions:
                msg = 'No runs found'
            else:
                msg = '. '.join(map(str, exceptions))
            raise ValueError(msg)
        elif exceptions:
            log( "WARNING: INVALID RUNS WILL BE EXCLUDED")
    else:
        pool = get_pool()
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
    log('returning from get_signal_many_parallel')
    return DataResult(signal, event_data)
    #return signal, event_data



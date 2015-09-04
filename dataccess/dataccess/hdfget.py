import Image
import glob
from numpy import *
import sys
import h5py


# Format string to generate the path to a given run's hdf5 file. Takes:
# (string, string, int) -> (expt name, expt name, run number)
HDF_NAME = "/reg/d/psdm/MEC/%s/hdf5/%s-r%04.i.h5"

# Background indices:
# Inputs:
#   run: a run number
# Output:
#   An list of indices of dark events
def background(run, expname):
    darkevents = []
    f=h5py.File(HDF_NAME%(expname, expname, run),'r')
    evrData=f['/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0/data']
    a = [evt['eventCode'] for evt in evrData['fifoEvents']]
    for i, evt in enumerate(a):
        if 162 in evt:
            # First run is always flagged and most of the time it shouldn't be.
            if i == 0:
                try:
                    if 162 in a[24]:
                        darkevents.append(i)
                except IndexError:
                    pass
            else:
                darkevents.append(i)
    return darkevents


# GetArray function:
# Gets a cspad image as an array from the hdf5 file.
# Pattern of CsPad chips determined from testing.py and comparison to outputs
# of the psana library's data import feature
# Input:
#   run: run number
#   event: event number (starts at 1)
# Outputs:
#   numpy array shape 388 x 370 or  830 x 825

def getImg(detid, run, expname):
    print "processing", HDF_NAME % (expname, expname, run)
    f=h5py.File(HDF_NAME % (expname, expname, run),'r')
    detectors = {1: '2x2::ElementV1/MecTargetChamber.0:Cspad2x2.1', 2: '2x2::ElementV1/MecTargetChamber.0:Cspad2x2.2', 3: '::ElementV2/MecTargetChamber.0:Cspad.0'}
    ref_string = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad%s/data'
    #print ref_string%detectors[detid]
    data = f[ref_string%detectors[detid]]
    nevents = len(data)
    eventlist = []
    for evt in data:
        if detid == 3:
            eventlist.append(proc_raw_quad_data(evt.astype('float64')))
        else:
            eventlist.append(proc_raw_single_data(evt.astype('float64')))
    return nevents, array(eventlist), background(run, expname)

def proc_raw_single_data(data):
    """
    Given a 185x388x2 array extracted from an hdf5 file, return the assembeled
    image data.
    """
    piece1, piece2 = rot90(data[:, :, 0], 1), rot90(data[:, :, 1], 1)
    concat = hstack((piece1, piece2))
    return insert(concat, (194), zeros((3, len(concat[0]))), axis = 0)

def proc_raw_quad_data(data):
    """
    Given an 8x185x388 array extracted from an hdf5 file, return the assembeled
    image data.
    """
    output = zeros((830,825))
    corners = [
        [429,421],
        [430,634],
        [420,1],
        [633,0],
        [0,213],
        [0,1],
        [16,424],
        [228,424]
        ]
    rotated = [1,1,0,0,3,3,0,0]
    for i, arr in enumerate(data[event-1]):
        a = rot90(insert(arr,(193,193,193,193),0,axis = 1),rotated[i])
        if rotated[i]:
            output[corners[i][0]:corners[i][0]+392, corners[i][1]:corners[i][1]+185] = a
        else:
            output[corners[i][0]:corners[i][0]+185, corners[i][1]:corners[i][1]+392] = a
    return output
    

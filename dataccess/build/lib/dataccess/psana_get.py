# Author: O. Hoidn

import pdb
import sys
import os

from psana import *
import numpy as np

import utils
import config


# TODO: rename this module so that it's clear that it's only used with the deprecated version
# of the python psana API
# psana configuration
configFileName = utils.resource_path('data/tiff_converter_dump_epics.cfg')
print configFileName
assert os.path.exists(configFileName), "Config file not found, looked for: %s" \
       % configFileName
setConfigFile(configFileName)


def getImg(detid, run):
    """
    TODO
    """
    DIVERTED_CODE = 162
    ds = DataSource('exp=%s:run=%d:stream=0,1'% (config.expname,run) )
    # TODO: throw exception if key missing from config
    make_detector_source = lambda key: Source('DetInfo(' + config.detinfo_map[key].device_name + ')')
    det = make_detector_source(detid)
    nevent = 0
    arraylist = []
    darkevents = []
    exposures = []
    codes = []
    for i, evt in enumerate(ds.events()):
        # get the event codes
        image = evt.get(EvrData.DataV3, Source('DetInfo(NoDetector.0:Evr.0)'))
        for fifoEvent in image.fifoEvents():
            # In general, this will incorrectly add i = 0 to darkevents
            if fifoEvent.eventCode() == DIVERTED_CODE:
                darkevents.append(i)
            else:
                exposures.append(i)
        nevent+=1
        calibframe = evt.get(ndarray_int16_2, det, 'image0')
        #calibframe =  det.image(evt)
        if calibframe is not None:
            #if det.__str__() == 'Source("DetInfo(MecTargetChamber.0:Cspad.0)")':
            # detector is a quad CSPAD
            if config.detinfo_map[detid].dimensions == (830, 825):
                cropframe = calibframe[70:900,0:825].astype(float)
            else:
                cropframe = calibframe.astype(float)
        else:
            print 'this event does not contain %s' % det.__str__()
            continue        
        #print cropframe.shape
        arraylist.append(cropframe)
    # In 120 Hz mode, the first event is likely to be incorrectly flagged as
    # dark. We correct this here.
    if darkevents[0] == 0:
        if darkevents[1] != 24:
            exposures = [darkevents.pop(0)] + exposures
    return nevent, np.array(arraylist), darkevents


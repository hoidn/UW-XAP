
#!/usr/bin/env python
# Script to generate calibrated tiff images for cspad's
# and dump a list of EPICS variables to a TXT file including photon energy
# Author: Zhou Xing
# Email: zxing@slac.stanford.edu

import pdb
import sys
import os
import numpy as np
import gc
#from AngularIntegrationM import AngularIntegratorM


#if len(sys.argv)!=3:
#    print 'Syntax: tiff_converter_dump_epics.py <run> <exp. name>'
#    sys.exit()

saveImage = True

from psana import *
import Image

sys.path.insert(1, '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
import utils


basedir = os.path.split(os.path.abspath( __file__ ))[0]
configFileName = os.path.join(basedir,"tiff_converter_dump_epics.cfg")
print configFileName
assert os.path.exists(configFileName), "Config file not found, looked for: %s" \
       % configFileName
setConfigFile(configFileName)



#@utils.persist_to_file("cache/getImg.p")
def getImg(det_label, run, expname):
    gc.collect() # collect garbage
    """
    det == 1, 2, or 3
    """
    # event code for events that were diverted to endstation A
    DIVERTED_CODE = 162
    ds = DataSource('exp=MEC/%s:run=%d:stream=0,1'% (expname,run) )
    src = [ Source('DetInfo(MecTargetChamber.0:Cspad.0)') ]
    src += [ Source('DetInfo(MecTargetChamber.0:Cspad2x2.1)') ]
    src += [ Source('DetInfo(MecTargetChamber.0:Cspad2x2.2)') ]
    src += [ Source('DetInfo(MecTargetChamber.0:Cspad2x2.3)') ]
    src += [ Source('DetInfo(MecTargetChamber.0:Cspad2x2.4)') ]
    detectors = {3: src[0], 1: src[1], 2: src[2]}
    det = detectors[det_label]
    nevent = 0
    arraylist = []
    blank_indices = []
    for i, evt in enumerate(ds.events()):
        nevent+=1
        calibframe = evt.get(ndarray_int16_2, det, 'image0')
        if calibframe is not None:
            if det.__str__() == 'Source("DetInfo(MecTargetChamber.0:Cspad.0)")':
                cropframe = calibframe[70:900,0:825].astype(float)
            else:
                cropframe = calibframe.astype(float)
        else:
            print 'this event does not contain %s' % det.__str__()
            continue        
        print cropframe.shape
        arraylist.append(cropframe)

        # determine if this event is a blank frame.
        image = evt.get(EvrData.DataV3, Source('DetInfo(NoDetector.0:Evr.0)'))
        for fifoEvent in image.fifoEvents():
            if fifoEvent.eventCode()==DIVERTED_CODE:
                blank_indices.append(i)

        #print cropframe
        #im = Image.fromarray(cropframe)
        #file = '/reg/d/psdm/mec/%s/scratch/run_%d_evt_%d_%s.tif'%( expname,run,nevent,det.__str__())
        #if saveImage:
        #    print 'writing file',file
        #    im.save(file)
    return nevent, np.array(arraylist), blank_indices


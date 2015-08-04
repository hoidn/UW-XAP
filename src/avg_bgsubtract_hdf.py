#import ipdb
import sys
import math
import numpy as np
import Image
import glob
import argparse
import os
import hdfget

#TODO: blankimgs and outliers should be disjoint

EXPNAME = "mecd6714"
DIMENSIONS_DICT = {1: (400, 400), 2: (400, 400), 3: (830, 825)}

def fileNames(runNum, detid):
    if detid in [1, 2]:
        return glob.glob(BASEDIR + "/spectrometers/r00" +  str(runNum) + "/CS" + str(detid) + "/run_" + str(runNum) + "*.tif")
    elif detid == 3: 
        return glob.glob(BASEDIR + "/quad/r00" +  str(runNum) + "/run_" + str(runNum) + "*.tif")
    else:
        sys.exit("invalid detector id")

def findDark(runNum, detid, firstblk = None):
    nfiles, eventlist = hdfget.getImg(detid, runNum)
    print "run", runNum, "XRTS detector", detid
    print str(nfiles) + " tifs found"
    
    if nfiles == 0:
        sys.exit("no files found")
    
    if nfiles >= 25:
        totalcounts = []
        for event in eventlist:
            event  = np.array(im)
            total_signal = np.sum(event)
            totalcounts.append(total_signal)
        
        cutoff = (np.mean(totalcounts) + np.std(totalcounts), np.mean(totalcounts) - np.std(totalcounts))
        outliers = filter(lambda i: totalcounts[i] > cutoff[0] or totalcounts[i] < cutoff[1], range(nfiles))
        print len(outliers), "outliers at 1 sigma"
        
        if firstblk != None:
            blankn = firstblk - 1
        tmp = totalcounts[:]
        tmp.sort()
        blankimgs = filter(lambda i: totalcounts[i] <= tmp[nfiles/24], range(nfiles))
        checkspacing = [ (map(lambda x: (x-b)%24, blankimgs)).count(0) for b in blankimgs ]
        blankn = ( blankimgs[checkspacing.index(max(checkspacing))] )%24
        
        blankimgs = range(blankn, nfiles, 24)
        print "blank images:", blankimgs
        return {'blanks': blankimgs, 'outliers': outliers, 'nfiles': nfiles, 'totalcounts': totalcounts, 'events': eventlist}
    else: 
        print "no background frames processed"
        return {'blanks': [], 'outliers': [], 'nfiles': 1, 'events': None}

def avgSignalAndDark(infoDict, runNum, detid):
    """
    given a list of blank shot numbers within a run, separately average 
    the blank and non-blank exposures for detector corresponding to detid 
    separately.

    args: 
        infoDict: result from findDark
        runNum, detid: slf-explanatory

        returns a dict containing the frames, with keys 'signal' and 
            'background'
    """
    #ipdb.set_trace()
    nfiles = infoDict['nfiles']
    outliers = infoDict['outliers']
    blankimgs = infoDict['blanks']
    events = infoDict['events']
    bg = np.zeros(DIMENSIONS_DICT[detid])
    signal = np.zeros(DIMENSIONS_DICT[detid]) 

    for b in blankimgs:
        imarray = events[b]
        bg += imarray
    if len(blankimgs) > 0:
        bg /= float(len(blankimgs))
    
    for n in range(nfiles):
        if n not in blankimgs and n not in outliers:
            imarray = events[n]
            signal += imarray
    signal /= float(nfiles - len(blankimgs) - len(outliers))
    os.system('mkdir -p bgsubtracted_120Hz')
    np.savetxt("./bgsubtracted_120Hz/r00" + str(runNum) + "_" + str(detid) + "_signal.dat", signal)
    np.savetxt("./bgsubtracted_120Hz/r00" + str(runNum) + "_" + str(detid) + "_bg.dat", bg)
    np.savetxt("./bgsubtracted_120Hz/r00" + str(runNum) + "_" + str(detid) + "_bg_single.dat", signal - bg)
    return {'signal': signal, 'background': bg}

def avgMany(infoDicts, runList, detid):
    """
    average blank and signal frames over multiple runs
    """
    #first and last runs
    boundaries = map(str, [runList[0], runList[-1]])
    avgDicts = [avgSignalAndDark(infoDicts[i], runList[i], detid) for i in range(len(runList))]
    blanks = np.array([dict['background'] for dict in avgDicts])
    signal = np.array([dict['signal'] for dict in avgDicts])
    signal = np.mean(signal, axis = 0)
    if np.shape(blanks[0]) != np.shape(signal[0]):
        blanks = np.zeros(np.shape(signal[0]))
    else:
        blanks = np.mean(blanks, axis = 0)

    os.system('mkdir -p bgsubtracted_120Hz/summed')
    np.savetxt("./bgsubtracted_120Hz/summed" + "-".join(boundaries) + "_" + str(detid) + "_bg.dat", blanks)   
    np.savetxt("./bgsubtracted_120Hz/summed" + "-".join(boundaries) + "_" + str(detid) + "_signal.dat", signal)
    np.savetxt("./bgsubtracted_120Hz/summed" + "-".join(boundaries) + "_" + str(detid) + "_bgsubbed.dat", signal - blanks)   
    return {'signal': signal, 'background': blanks}
    

def main(runList):
    #use the quad exposures to find blank shots
    blankInfo = [findDark(runNum, 3) for runNum in runList]
    for detid in [1,2,3]: 
        avgMany(blankInfo, runList, detid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type = int, nargs = '+',  help = 'start and end numbers of ranges of runs to process')
    parser.add_argument('--sixty', '-s', action = 'store_true',  help = 'data run taken at 60 Hz rather than 120')

    args = parser.parse_args()
    if len(args.run)%2 != 0:
        raise ValueError("number of args must be positive and even (i.e., must specify one or more ranges of runs)")
    fullRange = []
    for i in range(len(args.run)/2):
        fullRange += range(args.run[2 * i], 1 + args.run[2 * i + 1])
    main(fullRange)


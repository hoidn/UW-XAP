import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
#import ipdb
import matplotlib as mp
import os

import data_access as data
import pdb


#set matplotlib linewidth parameters
#mp.rcParams['axes.linewidth'] = .3
#mp.rcParams['lines.linewidth'] = .3
#mp.rcParams['patch.linewidth'] = .3

# backsatter energy for graphite 002
e0 = 1848.303414

# elements and their emission energies
elements = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni"]
emission = [
    [4090.6,4086.1,4460.5,4492.0],      #Sc
    [4510.84,4504.86,4931.81,4966.0],   #Ti
    [4952.2,4944.64,5427.29,5465.0],    #V
    [5414.72,5405.509,5946.71,5989.0],  #Cr
    [5898.75,5887.65,6490.45,6539.0],   #Mn
    [6403.84,6390.84,7057.98,7112.0],   #Fe
    [6930.32,6915.3,7649.43,7709.0],    #Co
    [7478.15,7460.89,8264.66,8333.0]    #Ni
    ]
lineidforplot = ["$K\\alpha_1$","$K\\alpha_2$","$K\\beta_{1,3}$","$E_F$"]

def center_col(label, detid):
    """
    Return the peak index of the label data summed along the zeroth axis
    (perpendicular to the energy-dispersive direction
    """
    summed = np.sum(data.get_label_data(label, detid), axis = 0)
    return np.argmax(summed)

def lineout(label, detid, cencol, pxwidth = 10, **kwargs):
    """
    Return a 1d lineout
    """
    raw = data.get_label_data(label, detid, **kwargs)
    spectrum_intensities = np.array([ sum( [raw[i][j] for j in range(cencol-pxwidth,cencol+pxwidth+1)] ) for i in range(400) ])
    return spectrum_intensities

def get_normalization(spectrum, sumwidth = 40):
    nalpha = np.argmax(spectrum)
    print "peak index: ", nalpha
    return np.sum(spectrum[max(nalpha-sumwidth,0):min(nalpha+sumwidth,len(spectrum) - 1)])

def get_energies(label, detid, eltname, cencol, **kwargs):
    """
    Return 1d array of energies corresponding to rows on the detector 
    based on calibration off of the ka and kb positions in XES data of the 
    given label and detid (1 or 2)

    Returns a tuple:
    Row index of the k alpha peak, 1d array of energies
    """
    frame_dimension = 400
    if detid not in [1, 2]:
        raise ValueError("Invalid detector id for von Hamos spectrometer")

    # element id
    eltid = elements.index(eltname)
    kalpha = emission[eltid][0]
    kbeta = emission[eltid][2]

    spectrum = lineout(label, detid, cencol, **kwargs)[::-1]
    nalpha = np.argmax(spectrum)
    offset = nalpha + 20
    nbeta = np.argmax(spectrum[offset:]) + offset

    # calculate position of peak positions on spectrometer
    thalpha = math.asin(e0/kalpha)
    posalpha = 103.4/(math.tan(thalpha))
    thbeta = math.asin(e0/kbeta)
    posbeta = 103.4/(math.tan(thbeta))

    # calculate pixel size
    pxsize = (posbeta - posalpha)/(nalpha - nbeta)

    # calculate pixel positions
    pixels = range(nalpha,nalpha-frame_dimension,-1)
    pixels = [ posalpha + pxsize*n for n in pixels ]

    # calculate Bragg angles and energies for graphite 002
    thetalist = [ math.atan(103.4/p) for p in pixels ]
    elist = [ 1848.303414/(math.sin(theta)) for theta in thetalist ]

    nrm = np.sum(spectrum[max(nalpha-40,0):min(nalpha+40,frame_dimension)])
    return np.array(elist[::-1])

def get_spectrum_one_label(eltname, detid, label, cold_calibration_label, \
        cencol_calibration_label, pxwidth = 10, default_bg = None):
    cencol = center_col(cencol_calibration_label, detid)
    energies = get_energies(cold_calibration_label, detid, eltname, cencol)
    intensities = lineout(label, detid, cencol, pxwidth = pxwidth, default_bg = default_bg)
    norm = get_normalization(intensities)
    return energies, intensities / norm

def main(eltname, detid, labels, cold_calibration_label, \
        cencol_calibration_label, pxwidth = 10, default_bg = None):
    spectrumList = []
    if not os.path.exists('xes/'):
        os.makedirs('xes')
    for label in labels:
        energies, intensities = get_spectrum_one_label(eltname, detid, label,\
            cold_calibration_label, cencol_calibration_label, pxwidth = pxwidth, default_bg = default_bg)
        spectrumList.append([energies, intensities])
        np.savetxt('xes/' + label + '_' + eltname + '_' + str(detid),
[energies, intensities], header = 'energy (eV)\tintensity (arb)')
    name = 'plots_xes/' + '_'.join(labels) + '_' + eltname + '_' + str(detid)
    plot_spectra(eltname, spectrumList, labels, name)


def plot_spectra(eltname, spectrumList, labels, name = None):
    if not os.path.exists('plots_xes/'):
        os.makedirs('plots_xes/')
    eltid = elements.index(eltname)
    elist = spectrumList[0][0]
    max_intensity = np.max(map(lambda x: x[1], spectrumList))
    plt.plot(elist, spectrumList[0][1], label = labels[0])
    plt.axhline(y=0, color = "black")
    plt.title(elements[eltid] + " XES")

    #add vertical lines to identify peaks within observed range
    txtshift = [0, -20, -25, 0]
    txtheight = [1.1, 1.1, 0.5, 0.5]
    for i in range(4):
        if elist[-1] - 50 < emission[eltid][i] < elist[0] + 50:
            plt.plot( [emission[eltid][i], emission[eltid][i]],
                [(-0.05)*max_intensity, (txtheight[i])*max_intensity],
                color = "gray")
            plt.text(emission[eltid][i]+txtshift[i], (txtheight[i])*max_intensity, lineidforplot[i], size="large")

    colorlist = 4 * ["orange", "green", "purple", "red", "brown", "black"]
    ncolors = len(colorlist)

    for spectrum, label, n in zip(spectrumList[1:], labels[1:], range(len(labels[1:]))):
        plt.plot(spectrum[0], spectrum[1], label = label, color = colorlist[(n-1)])

    plt.legend()
    plt.xlabel("Energy (eV)", size="large")
    plt.ylabel("Counts", size="large")
    plt.ylim((0, 1.15 * max_intensity))
    if name:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.savefig(name + '.svg', bbox_inches='tight')
    plt.show()


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eltname', help = 'element name')
    parser.add_argument('calibration', type = str, help = 'name of group of cold runs to use for calibration of the energy scale and identification of the subregion of the CSPAD to process into a spectrum')
    parser.add_argument('detid', type = int, help = 'Detector ID')
    parser.add_argument('--pxwidth', '-p', type = int, default = 10, help = 'pixel width of CSPAD subregion to sum')
    parser.add_argument('--background', '-b',  help = 'Use runs of this label for background subtraction instead of extracting dark exposures from the run if interposed background frames are absent. \nThis is necessary for 60 Hz runs.')
    parser.add_argument('datalabels', nargs = '+', help = 'Labels of run groups to process.')
    args = parser.parse_args()
    eltname = args.eltname
    cold_calibration_label = args.calibration
    cencol_calibration_label = args.calibration
    pxwidth = args.pxwidth
    default_bg = args.background
    detid = args.detid
    labels = args.datalabels
    pxwidth = args.pxwidth
    main(eltname, detid, labels, cold_calibration_label, \
        cencol_calibration_label, pxwidth = pxwidth, default_bg = default_bg)

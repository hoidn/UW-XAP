import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mp
import os
from scipy import interpolate

import xes_energies

import pdb
from scipy.ndimage.filters import gaussian_filter as filt

data_extractor = np.genfromtxt

# backsatter energy for graphite 002
e0 = 1848.303414

# distance in mm between HOPG crystal and the spectrometer's
# source-to-dectector axis; i.e., the crystal's curvature radius
hopg_radius = 103.4

# Dict of emission line energies. 
emission = xes_energies.emission_dict()

# Maps emission line keys to plot labels
lineidforplot = {'ka1': "$K\\alpha_1$", 'ka2': "$K\\alpha_2$", 'kb': "$K\\beta_{1,3}$", 'Ef': "$E_F$"}

def center_col(data):
    """
    Return the peak index of the CSPAD data summed along the zeroth axis
    (perpendicular to the energy-dispersive direction
    """
    summed = np.sum(data, axis = 0)
    return np.argmax(summed)

def lineout(data, cencol, pxwidth = 3):
    """
    Return a 1d lineout
    """
    spectrum_num_points = len(data)
    spectrum_intensities = np.array([ sum( [data[i][j] for j in range(cencol-pxwidth,cencol+pxwidth+1)] ) for i in range(spectrum_num_points) ])
    return spectrum_intensities

def get_normalization(x, intensities, sumwidth = 150):
    n_ref1 = np.argmax(intensities)
    x_ref1 = x[n_ref1]
    filtered = intensities[np.logical_and(x > x_ref1 - sumwidth, x < x_ref1 + sumwidth)]
    #print "peak index: ", n_ref1
    return np.sum(filtered)

#def save_calib(fname, energies):
#    with open(fname, 'wb') as f:
#        np.savetxt(f, np.array([range(len(energies)), energies]).T, header = 'row index\tenergy(eV)')

def save_calib(spectrum_num_points, save_path, energy_ref1, energy_ref2, n_ref1, n_ref2):
    """
    energy_ref1: first (lower) reference energy
    energy_ref2: second (higer) reference energy
    n_ref1: array index corresponding to energy_ref1
    n_ref2: array index corresponding to energy_ref2

    Saves the provided constants to save_path
    """
    with open(save_path, 'wb') as f:
        np.savetxt(f, np.array([[spectrum_num_points], [energy_ref1], [energy_ref2], [n_ref1], [n_ref2]]).T, '%d\t %f\t %f\t %d\t %d', header = 'Number of points in spectrum\tEnergy 1 (eV)\tEnergy 2 (eV)\tpixel index 1\tpixel index 2')

#def load_calib(fname):
#    with open(fname, 'rb') as f:
#        energies =(np.genfromtxt(f).T)[1]
#    return energies

def load_calib(fname):
    def parse_config_objs(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2):
        return int(spectrum_num_points), float(energy_ref1), float(energy_ref2), int(n_ref1), int(n_ref2)

    with open(fname, 'rb') as f:
        spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2 = parse_config_objs(*np.genfromtxt(f).T)

    return energies_from_two_points(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2)

def get_k_energies_and_positions(eltname, spectrum):
    """
    Return the energies and indices of the k alpha and k beta peaks in
    spectrum.
    Arguments:
        spectrum: a 1d-array
    
    It is assumed that the largest two peaks in spectrum are the k alpha
    and k beta peak of a single element
    """
    try:
        energy_kalpha = emission[eltname]['ka1']
        energy_kbeta = emission[eltname]['kb']
    except KeyError:
        raise KeyError("element identifier not found: " + eltname)

    n_kalpha = np.argmax(spectrum)
    offset = n_kalpha + 20
    n_kbeta = np.argmax(spectrum[offset:]) + offset

    return energy_kalpha, energy_kbeta, n_kalpha, n_kbeta


def energies_from_two_points(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2):
    """
    Calculate an array of energy values corresponding to pixel indices using
    two reference points
    """
    # calculate position of peak positions on spectrometer
    thalpha = math.asin(e0/energy_ref1)
    posalpha = hopg_radius/(math.tan(thalpha))
    thbeta = math.asin(e0/energy_ref2)
    posbeta = hopg_radius/(math.tan(thbeta))

    # calculate pixel size
    pxsize = (posbeta - posalpha)/(n_ref1 - n_ref2)

    # calculate pixel horizontal positions relative to source point
    pixels = range(n_ref1-spectrum_num_points, n_ref1)
    pixels = [ posalpha + pxsize*n for n in pixels ]

    # calculate Bragg angles and energies for graphite 002
    thetalist = [ math.atan(hopg_radius/p) for p in pixels ]
    elist = [ e0/(math.sin(theta)) for theta in thetalist ]

    return elist

def energies_from_data(data, cencol, save_path = None, eltname = '', calibration_mode = 'k alpha k beta', **kwargs):
    """
    Return 1d array of energies corresponding to rows on the detector 
    based on calibration off of the ka and kb positions in XES data of the 
    given data

    If a string is assigned to save_path, the calibration is saved under
        that given name in the directory calibs/

    Returns a tuple:
    Row index of the k alpha peak, 1d array of energies
    """

    spectrum = lineout(data, cencol)[::-1]
    spectrum_num_points = len(spectrum)

    if calibration_mode == 'k alpha k beta':
        energy_ref1, energy_ref2, n_ref1, n_ref2 = get_k_energies_and_positions(eltname, spectrum)
    # else:....
    # TODO: add other energy calibration modes

    energies = energies_from_two_points(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2)

    nrm = np.sum(spectrum[max(n_ref1-40,0):min(n_ref1+40, spectrum_num_points)])
    #energies = elist[::-1]
    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname and (not os.path.exists(dirname)):
            os.system('mkdir -p ' + os.path.dirname(save_path))
        #save_calib(save_path, energies)
        save_calib(spectrum_num_points, save_path, energy_ref1, energy_ref2, n_ref1, n_ref2)
    return np.array(energies)

# TODO: allow masking out bad data ranges (probably something to put into
#  a config file) for background subtraction purposes
def get_spectrum(data, dark = None, cencol_calibration_data = None, cold_calibration_data = None,
        pxwidth = 3, bg_sub = True, calib_load_path = None, calib_save_path = None,
        energy_ref1_energy_ref2_calibration = True, eltname = ''):
    """
    Return the XES spectrum corresponding to the given data
    and element

    Inputs:
        eltname: string of the abbreviated element name
        data: 2d array of CSPAD data
        cold_calibration_data: data to use for determination of the energy
            scale. If None, the first argument is used for this. 
        cencol_calibration_data: data to use for location of the
            spectrometer's line of focus. If None, the first argument is used
            for this.
        pxwidth: width of the CSPAD lineout from which the spectrum is
            constructed
        peak_width:
            TODO: deprecate or not?
        bg_sub: if True, perform a constant subtraction. The subtracted
            constant is the 5th percentile of the spectrum after smoothing
            with a gaussian kernel of standard deviation 5
        calib_load_path: path to a file with an energy calibration to load
        calib_save_path: File to which to save an energy calibration if
            calib_load_path is None
        energy_ref1_energy_ref2_calibration: If calib_load_path is None, use k alpha
            and k beta peak locations to determine an energy scale. If None
            and calib_load_path is also None, do not perform an energy
            calibration at all.
    Output: array, array -> energy or index, normalized intensity
    """
    if np.shape(data) != (391, 370):
        print "WARNING: array dimensions differ from those of CSPAD140k"
    if energy_ref1_energy_ref2_calibration or calib_load_path:
        peak_width = 150
    else:
        peak_width = 15
    if dark is not None:
        # copy without mutating the original array
        data = np.array(data, copy = True) - dark
    if cencol_calibration_data is None:
        cencol_calibration_data = data
    cencol = center_col(cencol_calibration_data)
    intensities = lineout(data, cencol, pxwidth = pxwidth)
#    if calib_load_path:
#        x = load_calib(calib_load_path)
    if calib_load_path:
        x = load_calib(calib_load_path)
    elif energy_ref1_energy_ref2_calibration and eltname and (cold_calibration_data is not None):
        x = energies_from_data(cold_calibration_data, cencol, save_path = calib_save_path, eltname = eltname)
    else:
        if energy_ref1_energy_ref2_calibration and not eltname:
            print "No element identifier provided; skipping energy calibration."
        elif energy_ref1_energy_ref2_calibration and not cold_calibration_data:
            print "No file for calibration specified; skipping energy calibration"
        x = np.array(range(len(intensities)))
    if bg_sub:
        smoothed = filt(intensities, 5)
        floor = np.percentile(smoothed, 5)
        intensities -= floor
    norm = get_normalization(x, intensities, peak_width)
    return x, intensities / norm

def main(paths, cold_calibration_path = None, pxwidth = 3,
        calib_load_path = None, calib_save_path = None,
        dark_path = None, energy_ref1_energy_ref2_calibration = True,
        eltname = ''):
    spectrumList = []
    scale_ev = (energy_ref1_energy_ref2_calibration or calib_load_path)
    if not os.path.exists('xes_spectra/'):
        os.makedirs('xes_spectra')
    if cold_calibration_path:
        cold_calibration_data = data_extractor(cold_calibration_path)
    else:
        cold_calibration_data = None
    if dark_path:
        dark = data_extractor(dark_path)
    else:
        dark = None
    data_arrays = map(data_extractor, paths)
    labels = map(os.path.basename, paths)
    for data, label in zip(data_arrays, labels):
        energies, intensities = get_spectrum(data,
            cencol_calibration_data = data,
            dark = dark, cold_calibration_data = cold_calibration_data,
            pxwidth = pxwidth, calib_load_path = calib_load_path,
            calib_save_path = calib_save_path,
            energy_ref1_energy_ref2_calibration = energy_ref1_energy_ref2_calibration, eltname = eltname)
        spectrumList.append([energies, intensities])
        if eltname:
            np.savetxt('xes_spectra/' + label + '_' + eltname,
                [energies, intensities], header = 'energy (eV)\tintensity (arb)')
        else:
            np.savetxt('xes_spectra/' + label,
                [energies, intensities], header = 'energy (eV)\tintensity (arb)')
    if eltname:
        name = 'plots_xes/' + '_'.join(labels) + '_' + eltname
    else:
        name = 'plots_xes/' + '_'.join(labels)
    plot_spectra(spectrumList, labels, scale_ev, name = name, eltname = eltname)


def plot_spectra(spectrumList, labels, scale_ev, name = None, eltname = ''):
    if not os.path.exists('plots_xes/'):
        os.makedirs('plots_xes/')
    elist = spectrumList[0][0]
    max_intensity = np.max(map(lambda x: x[1], spectrumList))
    plt.plot(elist, spectrumList[0][1], label = labels[0])
    plt.axhline(y=0, color = "black")
    plt.title(eltname + " XES")

    #add vertical lines to identify peaks within observed range
    txtshift = {'ka1': 0, 'ka2': -20, 'kb': -25, 'Ef': 0}
    txtheight = {'ka1': 1.1, 'ka2': 1.1, 'kb': 0.5, 'Ef': 0.5}
    if eltname:
        lines = emission[eltname].keys()
        for line in lines:
            if elist[-1] - 50 < emission[eltname][line] < elist[0] + 50:
                plt.plot( [emission[eltname][line], emission[eltname][line]],
                    [(-0.05)*max_intensity, (txtheight[line])*max_intensity],
                    color = "gray")
                plt.text(emission[eltname][line]+txtshift[line], (txtheight[line])*max_intensity, lineidforplot[line], size="large")

    colorlist = 4 * ["orange", "green", "purple", "red", "brown", "black"]
    ncolors = len(colorlist)

    for spectrum, label, n in zip(spectrumList[1:], labels[1:], range(len(labels[1:]))):
        plt.plot(spectrum[0], spectrum[1], label = label, color = colorlist[(n-1)])

    plt.legend()
    if scale_ev:
        plt.xlabel("Energy (eV)", size="large")
    else:
        plt.xlabel("CSPAD index", size="large")
    plt.ylabel("Counts", size="large")
    plt.ylim((0, 1.15 * max_intensity))
    if name:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.savefig(name + '.svg', bbox_inches='tight')
    plt.show()

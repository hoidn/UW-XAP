# Authors: O. Hoidn, J. Pacold, and R. Valenza

import sys
import argparse
import numpy as np
import math
import matplotlib as mp
import os
import pandas as pd
from scipy import interpolate
import functools

import data_access as data
import utils
import data_access
import logbook
import playback

import pdb
from scipy.ndimage.filters import gaussian_filter as filt
from output import log

import config
if config.plotting_mode == 'notebook':
    from dataccess.mpl_plotly import plt
else:
    import matplotlib.pyplot as plt

# TODO: use the same data extractor as in xrd.py

# backsatter energy in eV for graphite 002
e0 = 1848.303414

# Twice the distance in mm between HOPG crystal and the spectrometer's
# source-to-dectector axis; i.e., the crystal's curvature diameter
hopg_diameter = 103.4


# Maps emission line keys to plot labels
lineidforplot = {'ka1': "$K\\alpha_1$", 'ka2': "$K\\alpha_2$", 'kb': "$K\\beta_{1,3}$", 'Ef': "$E_F$"}

# row format for XES table:
# Ele.  A   Trans.  Theory (eV) Unc. (eV)   Direct (eV) Unc. (eV)   Blend  Ef
# Data source: http://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
with open(utils.resource_path('data/fluorescence.txt'), 'rb') as f:
    tabdata = pd.read_csv(f, sep = '\t')

def bgsubtract_linear_interpolation(arr1d, average_length = 100):
    """
    Interpolate a line using the mean of the first and last average_length
    points. Subtract this from arr1d and return the result.
    """
    x = np.array(range(len(arr1d)))
    fl = (arr1d[0] + arr1d[-1])/2
    xlow, ylow = x[:average_length], np.repeat(np.mean(arr1d[:average_length]), average_length)
    xhigh, yhigh = x[-average_length:], np.repeat(np.mean(arr1d[-average_length:]), average_length)
    xi, yi = np.concatenate((xlow, xhigh)), np.concatenate((ylow, yhigh))
    interpolation = interpolate.interp1d(xi, yi, bounds_error = False, fill_value = fl)
    return arr1d - interpolation(x)

@utils.memoize(timeout = None)
def emission_dict():
    """
    Returns a dict of dicts mapping element name and emission line label
    to photon energy in eV.

    The line label keys are: 'ka1', 'ka2', 'kb', and 'Ef' (Fermi energy)
    The element keys are 'Ne' through 'Fm'

    The energies used are from column 5 in fluorescence.txt. This data file
    currently contains complete data only for ka1, ka2, and kb. Ef energies
    for a few elements have been manually added.
    """
    line_dict = {}
    line_lookup = {'KL2': 'ka2', 'KL3': 'ka1', 'KM3': 'kb', 'Ef': 'Ef'}
    def process_one_row(row):
        name, line, energy = row[0], line_lookup[row[2]], row[5]
        elt_dict = line_dict.setdefault(name, {})
        elt_dict[line] = energy
    for i, row in tabdata.iterrows():
        process_one_row(row)
    return line_dict

#emission = emission_dict()

def mean_from_label(detid, transpose = False):
    """
    Input: detector ID

    Output: Function that takes a string reference to data runs and returns
    a 2D CSPAD data array corresponding to detid and the string reference.

    If event_data_getter is not none the function returns an array of event
    frames for the run corresponding to the dataset label. If the dataset
    contains more than 1 run an exception is raised.
    """
    def data_getter(label):
        arr = data.eval_dataset_and_filter(label, detid)[0]
        arr = arr.astype('float64')
        if transpose:
            return arr.T
        else:
            return arr
    return data_getter

def events_from_label(detid, event_indices = None,
        number_events = None, event_data_getter = None):
    if event_data_getter is None:
        def event_data_getter(x, **kwargs):
            return x
    def data_getter_events(label):
        """
        Returns:
        event data, run numbers
        event data : list of np.ndarray
        run numbers : list of str
        """

        def get_run_events_and_runstrings(rundict, event_indices):
            result = [rundict[k]
                for k in event_indices
                if k in rundict]
            utils.log("searching for events: " + str(event_indices))
            good_indices = filter(lambda k: k in rundict, event_indices)
            utils.log("found events: " + str(good_indices))
            assert len(result) > 0
            return result, map(lambda key, event_index: str(key) + '_' + str(event_index), len(result) * [k], good_indices)

        d = data.eval_dataset_and_filter(label, detid, event_data_getter = event_data_getter)[1]
        if event_indices:
            if len(d) > 1:
                raise ValueError("Invalid dataset label: %s. Label must refer to exactly one run." % label)
            result, run_strings = get_run_events_and_runstrings(d.values()[0], event_indices)
        elif number_events:
            indices =\
                [np.linspace(0, len(rundict), number_events, dtype = int)
                for rundict in d.values()]
            result, run_strings = reduce(lambda x, y: x + y, map(get_run_events_and_runstrings, d.values(), indices))
        else:
            raise ValueError("number_events and event_indices are both None")
        return result, run_strings
    return data_getter_events

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
    spectrum_intensities = np.sum(data[:,cencol - pxwidth:cencol + pxwidth + 1], axis = 1)
    return spectrum_intensities

def get_normalization(x, intensities, sumwidth = 150):
    n_ref1 = np.argmax(intensities)
    x_ref1 = x[n_ref1]
    filtered = intensities[np.logical_and(x > x_ref1 - sumwidth, x < x_ref1 + sumwidth)]
    #print "peak index: ", n_ref1
    return np.sum(filtered)


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


def load_calib(fname):
    def parse_config_objs(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2):
        return int(spectrum_num_points), float(energy_ref1), float(energy_ref2), int(n_ref1), int(n_ref2)

    with open(fname, 'rb') as f:
        spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2 = parse_config_objs(*np.genfromtxt(f).T)

    return energies_from_two_points(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2)

def mask_peaks_and_iterpolate(x, y, peak_ranges, avg_interval = 20.):
    interpolationx = []
    interpolationy = []
    for peakmin, peakmax in peak_ranges:
        indices_min = np.where(np.abs(x - peakmin) < avg_interval)[0]
        ymin = [np.mean(y[indices_min])] * len(indices_min)
        xmin = list(x[indices_min])

        indices_max = np.where(np.abs(x - peakmax) < avg_interval)[0]
        ymax = [np.mean(y[indices_max])] * len(indices_max)
        xmax = list(x[indices_max])

        interpolationx += (xmin + xmax)
        interpolationy += (ymin + ymax)
        
#        good_indices = np.where(np.logical_or(x < peakmin, x > peakmax))[0]
#        y = y[good_indices]
#        x = x[good_indices]
    return utils.extrap1d(interpolate.interp1d(np.array(interpolationx), np.array(interpolationy)))

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
        energy_kalpha = emission_dict()[eltname]['ka1']
        energy_kbeta = emission_dict()[eltname]['kb']
    except KeyError:
        raise KeyError("element identifier not found: " + eltname)

    n_kalpha = np.argmax(spectrum)
    offset = n_kalpha + 20
    n_kbeta = np.argmax(spectrum[offset:]) + offset

    return energy_kalpha, energy_kbeta, n_kalpha, n_kbeta


# TODO: temporarily using a linear energy calibration. Fix this.
def energies_from_two_points(spectrum_num_points, energy_ref1, energy_ref2, n_ref1, n_ref2):
    """
    Calculate an array of energy values corresponding to pixel indices using
    two reference points
    """
#    # calculate position of peak positions on spectrometer
#    thalpha = math.asin(e0/energy_ref1)
#    posalpha = hopg_diameter/(math.tan(thalpha))
#    thbeta = math.asin(e0/energy_ref2)
#    posbeta = hopg_diameter/(math.tan(thbeta))
#
#    # calculate pixel size
#    pxsize = (posbeta - posalpha)/(n_ref1 - n_ref2)
#
#    # calculate pixel horizontal positions relative to source point
#    pixels = range(n_ref1-spectrum_num_points, n_ref1)
#    pixels = [ posalpha + pxsize*n for n in pixels ]
#
#    # calculate Bragg angles and energies for graphite 002
#    thetalist = [ math.atan(hopg_diameter/p) for p in pixels ]
#    elist = [ e0/(math.sin(theta)) for theta in thetalist ]
#
#    return elist
    m = (energy_ref2 - energy_ref1) / (n_ref2 - n_ref1)
    start = energy_ref1 - n_ref1 * m
    end = energy_ref2 + (spectrum_num_points - n_ref2) * m
    if start > end:
        start, end = end, start
    return np.linspace(start, end, num = spectrum_num_points)

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
        save_calib(spectrum_num_points, save_path, energy_ref1, energy_ref2, n_ref1, n_ref2)
    return np.array(energies)

# TODO: allow masking out bad data ranges (probably something to put into
#  a config file) for background subtraction purposes
def get_spectrum(data, dark = None, cencol_calibration_data = None, cold_calibration_data = None,
        pxwidth = 3, bg_sub = True, calib_load_path = None, calib_save_path = None,
        energy_ref1_energy_ref2_calibration = True, eltname = '', normalization = True):
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
    if calib_load_path:
        x = load_calib(calib_load_path)
    elif energy_ref1_energy_ref2_calibration and eltname and (cold_calibration_data is not None):
        x = energies_from_data(cold_calibration_data, cencol, save_path = calib_save_path, eltname = eltname)
    else:
        if energy_ref1_energy_ref2_calibration and not eltname:
            log( "No element identifier provided; skipping energy calibration.")
        elif energy_ref1_energy_ref2_calibration and not cold_calibration_data:
            log( "No file for calibration specified; skipping energy calibration")
        x = np.array(range(len(intensities)))
    if bg_sub:
        smoothed = filt(intensities, 5)
        floor = np.percentile(smoothed, 5)
        intensities -= floor
    if normalization:
        norm = get_normalization(x, intensities, peak_width)
    else:
        norm = 1.
    return x, intensities / norm


@utils.ifplot
@playback.db_insert
def plot_spectra(spectrumList, labels, scale_ev, eltname = ''):
    if not os.path.exists('plots_xes/'):
        os.makedirs('plots_xes/')
    elist = spectrumList[0][0]
    max_intensity = np.max(map(lambda x: x[1], spectrumList))
    plt.plot(elist, spectrumList[0][1], label = labels[0])
    # TODO: fix this
    #plt.plot([0, 0], [0, max_intensity], color = 'black')
    #plt.axhline(y=0, color = "black")
    #plt.title(eltname + " XES")

    #add vertical lines to identify peaks within observed range
    txtshift = {'ka1': 0, 'ka2': -20, 'kb': -25, 'Ef': 0}
    txtheight = {'ka1': 1.1, 'ka2': 1.1, 'kb': 0.5, 'Ef': 0.5}
    if eltname:
        lines = emission_dict()[eltname].keys()
        for line in lines:
            if elist[-1] - 50 < emission_dict()[eltname][line] < elist[0] + 50:
                plt.plot( [emission_dict()[eltname][line], emission_dict()[eltname][line]],
                    [(-0.05)*max_intensity, (txtheight[line])*max_intensity],
                    color = "gray")
                # TODO: implement text annotations in mpl_plotly.
                #plt.text(emission_dict()[eltname][line]+txtshift[line], (txtheight[line])*max_intensity, lineidforplot[line], size="large")

    colorlist = 4 * ["orange", "green", "purple", "red", "brown", "black"]
    ncolors = len(colorlist)

    for spectrum, label, n in zip(spectrumList[1:], labels[1:], range(len(labels[1:]))):
        plt.plot(spectrum[0], spectrum[1], label = label, color = colorlist[(n-1)])

    plt.legend()
    if scale_ev:
        plt.xlabel("Energy (eV)")#, size="large")
    else:
        plt.xlabel("pixel index")#, size="large")
    plt.ylabel("Counts")#, size="large")
    # TODO: implement this for mpl_plotly
    # plt.ylim((0, 1.15 * max_intensity))
    if eltname:
        name = 'plots_xes/' + '_'.join(labels) + '_' + eltname
    else:
        name = 'plots_xes/' + '_'.join(labels)
    if name:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.savefig(name + '.svg', bbox_inches='tight')
    @utils.ifplot
    def show():
        plt.show()
    show()


def get_plotinfo_one_label(detid, label, get_spectrum_partial,
        event_indices = None, number_events = None, transpose = False):
    """
    Arguments:
    detid : str
            
    Returns sufficient info for plotting the spectrum/spectra corresponding to
    one dataset label. Each spectrum is either an average over a dataset or an
    individual event, depending on the values of number_events and event_indices.
    
    Returns:
        spectrumList, labels 
    spectrumList : list
        List of spectra in the format [x, y]
    labels : list of strings
        Legend labels associated with each spectrum.
    """
    # Extract data from labels. 
    data_extractor = mean_from_label(detid, transpose = transpose)

    def spectrum_extractor(arr, **kwargs):
        """
        Given area detector data, return a spectrum.
        """
        # Cast to float64 to avoid overflows.
        arr = arr.astype('float64')
        if transpose:
            arr = arr.T
        return get_spectrum_partial(arr, cencol_calibration_data = arr)

    # Get event-by-event spectra
    if (event_indices is not None) or (number_events is not None):
        spectrumList, run_strings = events_from_label(detid,
            event_indices = event_indices, number_events = number_events,
            event_data_getter = spectrum_extractor)(label)
        if event_indices is None:
            labels = run_strings
        else:
            labels =\
                [label + '_' + str(n)
                for n in event_indices]
    # Get a single averaged spectrum for the dataset label
    else:
        mean_arr = data.eval_dataset_and_filter(label, detid)[0]
        spectrumList = [spectrum_extractor(mean_arr)]
        return spectrumList, [label]
        
    return spectrumList, labels


def spectrum(detid, data_identifiers, event_indices = None, number_events = None,
        cold_calibration_label = None,
        pxwidth = 3, calib_load_path = None, calib_save_path = None,
        dark_label = None, energy_ref1_energy_ref2_calibration = True,
        eltname = '', transpose = False,
        normalization = True, bgsub = True):
    # Extract data from labels. 
    data_extractor = mean_from_label(detid, transpose = transpose)
    scale_ev = (energy_ref1_energy_ref2_calibration or calib_load_path)
    if cold_calibration_label:
        cold_calibration_data = data_extractor(cold_calibration_label)
    else:
        cold_calibration_data = None
    if dark_label:
        dark = data_extractor(dark_label)
    else:
        dark = None

    get_spectrum_partial = functools.partial(get_spectrum, dark = dark,
            cold_calibration_data = cold_calibration_data,
            pxwidth = pxwidth, calib_load_path = calib_load_path,
            calib_save_path = calib_save_path,
            energy_ref1_energy_ref2_calibration = energy_ref1_energy_ref2_calibration,
            eltname = eltname, normalization = normalization,
            bg_sub = bgsub)

    spectrumList, labels =\
        reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]),
            [get_plotinfo_one_label(detid, data_identifier, get_spectrum_partial,
            event_indices = event_indices, number_events = number_events, transpose = transpose)
            for data_identifier in data_identifiers])

    for label, spectrum in zip(labels, spectrumList):
        if eltname:
            path = 'spectra/' + label + '_' + eltname
        else:
            path = 'spectra/' + label
        if (event_indices is not None) or (number_events is not None):
            init_dict = {'event_label': label}
        else:
            init_dict = {}
        try:
            utils.save_data(spectrum[0], spectrum[1], path, init_dict = init_dict)
        except OSError, e:
            raise OSError("data directory 'spectra/' could not be created: %s" % str(e))

    plot_spectra(spectrumList, labels, scale_ev, eltname = eltname)
    return spectrumList


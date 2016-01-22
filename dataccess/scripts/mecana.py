#!/reg/g/psdm/sw/releases/ana-current/arch/x86_64-rhel7-gcc48-opt/bin/python2.7

# Author: O. Hoidn
import sys


sys.path.append('.') # so that config.py can be imported
import os
import argparse
from dataccess import utils

def addparser_init(subparsers):
    init = subparsers.add_parser('init', help =  'Initialize config.py in local directory.')
    # Unfortunate hack to figure out which subcommand was called. It looks
    # like the dest keyword for add_parser isn't usable in python 2.7.
    init.add_argument('--initcalled', help = 'dummy argument')

def call_init(config_source, config_dst):
    """
    Inputs:
        config_source, the path of this program's config file template.
        config_dst, path to which to copy the config file template

    Initializes config.py in the working directory
    """
    import shutil
    shutil.copyfile(config_source, config_dst)


def addparser_xes(subparsers):
    # XES analysis sub-command
    xes = subparsers.add_parser('spectrum', help = 'Process area detector data into spectra.')
    xes.add_argument('detid', type = str, help = 'Detector ID.')
    xes.add_argument('labels', nargs = '+', help = 'Labels of datasets to process into spectra.')
    xes.add_argument('--pxwidth', '-p', type = int, default = 3, help = 'Pixel width of CSPAD subregion to sum.')
    xes.add_argument('--rotate', '-r', action = 'store_true', help = "Toggles the area detector's axis of integration for generating spectra")
    xes.add_argument('--calibration', '-c', type = str, help = 'Label of dataset to use for calibration of the energy scale (if --energy_ref1_energy_ref2_calibration is selected but a calibration file is not provided). If not provided this parameter defaults to the first dataset in labels.')
    xes.add_argument('--subtraction', '-d', type = str, help = 'Label of dataset to use as a dark frame subtraction')
    xes.add_argument('--energy_ref1_energy_ref2_calibration', '-k', action = 'store_true', help = 'Enable automatic generation of energy calibration based on k alpha and k beta peak locations if --calibration_load_path is not given.')
    xes.add_argument('--eltname', '-e', default = '', help = 'Element name. This parameter is required for XES-based calibration; i.e., for generating an energy scale using ENERGY_REF1_ENERGY_REF2_CALIBRATION.')
    xes.add_argument('--calibration_save_path', '-s', type = str, help = 'Path to which to save energy calibration data if calibration_load_path is unspecified and --energy_ref1_energy_ref2_calibration is selected.')
    xes.add_argument('--calibration_load_path', '-l', type = str, help = 'Path from which to load energy calibration data.')

def call_xes(args):
    """
    Input: args, a value returned by argparse.ArgumentParser.parse_args()

    Calls the xes sub-command of this script.
    """
    from dataccess import xes_process
    labels = args.labels
    eltname = args.eltname
    if args.calibration:
        calibration = args.calibration
    else:
        calibration = labels[0]
    cold_calibration_label = calibration
    pxwidth = args.pxwidth
    pxwidth = args.pxwidth
    calibration_save_path = args.calibration_save_path
    calibration_load_path = args.calibration_load_path
    dark_label = args.subtraction
    energy_ref1_energy_ref2_calibration = args.energy_ref1_energy_ref2_calibration
    xes_process.main(args.detid, labels, cold_calibration_label = cold_calibration_label,
        pxwidth = pxwidth,
        calib_save_path = calibration_save_path, calib_load_path =
            calibration_load_path, dark_label = dark_label,
            energy_ref1_energy_ref2_calibration = energy_ref1_energy_ref2_calibration,
            eltname = eltname, transpose = args.rotate)

def addparser_xrd(subparsers):
    xrd = subparsers.add_parser('xrd', help = 'Process quad CSPAD data into powder patterns.')
    
    xrd.add_argument('detid', type = str, help = 'Detector ID.')
    xrd.add_argument('labels', nargs = '+', help = 'One or more dataset labels to process.')
    xrd.add_argument('--compounds', '-c', nargs = '+', help = 'Chemical formulas of crystalline species in the sample. If --background_subtraction is passed these MUST be provided.')
    xrd.add_argument('--background_subtraction', '-b', action = 'store_true', help = 'If selected, background subtraction will be performed by interpolation based on the signal between Bragg peaks.')
    xrd.add_argument('--peak_progression_compound', '-p', type = str, help = 'Compound for which to plot the progression of Bragg peak intensities as a function of incident flux if two or more datasets are being processed. If not specified, this option defaults to the first value of COMPOUNDS.')
    xrd.add_argument('--normalization', '-n', type = str, default = None, help = "Normalization option.\n\tIf == 'transmission', normalize by beam transmission specified in logbook;\n\tIf == 'background', normalize by background level (requires --compounds).\nBy default no normalization is applied to powder patterns, and peak progression plots are normalized by background level.")
    xrd.add_argument('--maxpeaks', '-m', type = int, default = None, help = "Limit the plot of peak intensities as a function of incident flux to the MAXPEAKS most intense ones")

def call_xrd(args):
    """
    Input: args, a value returned by argparse.ArgumentParser.parse_args()

    Calls the xrd sub-command of this script.
    """
    from dataccess import xrd
    detid = args.detid
    data_identifiers = args.labels
    mode = 'label'
    peak_progression_compound = args.peak_progression_compound
    bgsub = args.background_subtraction
    compound_list = args.compounds
    normalization = args.normalization
    if args.maxpeaks:
        maxpeaks = args.maxpeaks
    else:
        maxpeaks = 6
    xrd.main(detid, data_identifiers, mode = mode,
        peak_progression_compound = peak_progression_compound,
        bgsub = bgsub, compound_list = compound_list,
        normalization = normalization, maxpeaks = maxpeaks)

def addparser_histogram(subparsers):
    import numpy as np
    histogram = subparsers.add_parser('histogram', help =  'For a given dataset and detector ID, evaluate a function (defined in config.py) over all events and plot a histogram of the resulting values.')
    histogram.add_argument('detid', type = str, help = 'Detector ID.')
    histogram.add_argument('label', help = 'Label of dataset to process.')
    histogram.add_argument('--function', '-u', default = 'np.sum', help = 'Name of function (defined in config.py) to evaluate over all events. Defaults to np.sum.')
    histogram.add_argument('--filter', '-f', action = 'store_true', help = 'If selected, logbook-specified event filtering will be applied.')
    histogram.add_argument('--nbins', '-n', type = int, default = 100, help = 'Number of bins in the histogram.')

def call_histogram(args):
    from dataccess import summarymetrics
    label = args.label
    detid = args.detid
    nbins = args.nbins
    summarymetrics.main(label, detid, funcstr = args.function, filtered = args.filter, nbins = nbins)

def addparser_datashow(subparsers):
    datashow = subparsers.add_parser('datashow', help = 'For a given dataset and area detector ID, show the summed detector image and save it to a file in the working directory. Any detector masks specified in config.py can optionally be applied.')
    datashow.add_argument('detid', type = str, help = 'Detector ID.')
    datashow.add_argument('label', help = 'Label of dataset to process.')
    datashow.add_argument('--output', '-o', help = 'Path of output file')
    datashow.add_argument('--masks', '-m', action = 'store_true',
        help = 'Apply detector masks from config.py')
    datashow.add_argument('--max', '-a', type = int, help = "Maximum amplitude of color scale")

def addparser_eventframes(subparsers):
    datashow = subparsers.add_parser('eventframes', help = 'For a given dataset and area detector ID, show the summed detector image and save it to a file in the working directory. Any detector masks specified in config.py can optionally be applied.')
    datashow.add_argument('detid', type = str, help = 'Detector ID.')
    datashow.add_argument('label', help = 'Label of dataset to process.')
    datashow.add_argument('--output', '-o', help = 'Path of output file')
    datashow.add_argument('--masks', '-m', action = 'store_true',
        help = 'Apply detector masks from config.py')

def call_datashow(args):
    from dataccess import datashow
    label = args.label
    detid = args.detid
    rmax = args.max
    datashow.main(label, detid, args.output, masked = args.masks, rmax = rmax)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

# Add sub-commands to parser
addparser_init(subparsers)
addparser_xes(subparsers)
addparser_xrd(subparsers)
addparser_histogram(subparsers)
addparser_datashow(subparsers)

args = parser.parse_args()

config_source = utils.resource_path('data/config.py')
config_dst = 'config.py'
if 'initcalled' in args: # Enter init sub-command
    call_init(config_source, config_dst)
else:
    if not os.path.isfile(config_dst):
        parser.error("File config.py not found. Run 'python dataccess.py init' to initialize config.py, and then edit it appropriately before re-running")
    elif 'pxwidth' in args: # Enter xes sub-command
        call_xes(args)
    elif 'compounds' in args: # Enter xrd sub-command
        call_xrd(args)
    elif 'nbins' in args: # Enter histogram sub-command
        call_histogram(args)
    else: # Enter datashow sub-command
        call_datashow(args)

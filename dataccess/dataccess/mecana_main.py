import ipdb
import sys
#del sys.modules['pickle']
#sys.path.insert(0, '/reg/neh/home/ohoidn/anaconda2/lib/python2.7/')
sys.path.append('.') # so that config.py can be imported
import database
import playback
import argument_parsers
import utils
import argparse
import time
import config

def call_init(config_source, config_dst):
    """
    Inputs:
        config_source, the path of this program's config file template.
        config_dst, path to which to copy the config file template

    Initializes config.py in the working directory
    """
    import shutil
    shutil.copyfile(config_source, config_dst)

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
    normalization = not args.normalization
    bgsub = not args.nosubtraction
    calibration_save_path = args.calibration_save_path
    calibration_load_path = args.calibration_load_path
    dark_label = args.subtraction
    energy_ref1_energy_ref2_calibration = args.energy_ref1_energy_ref2_calibration
    if args.variation:
        # plot shot to shot variation
        # TODO: bring this up to date with addition of the --events option
        xes_process.main_variation(args.detid, labels, cold_calibration_label = cold_calibration_label,
            pxwidth = pxwidth,
            calib_save_path = calibration_save_path, calib_load_path =
                calibration_load_path, dark_label = dark_label,
                energy_ref1_energy_ref2_calibration = energy_ref1_energy_ref2_calibration,
                eltname = eltname, transpose = args.rotate, vn=args.variation_n, vc=args.variation_center,
                vs=args.variation_skip_width, vw=args.variation_width,
                normalization = normalization, bgsub = bgsub)
    else:
        # plot summed spectra
        xes_process.main(args.detid, labels, nevents = args.events,
            cold_calibration_label = cold_calibration_label, pxwidth = pxwidth,
            calib_save_path = calibration_save_path, calib_load_path =
                calibration_load_path, dark_label = dark_label,
                energy_ref1_energy_ref2_calibration = energy_ref1_energy_ref2_calibration,
                eltname = eltname, transpose = args.rotate,
                normalization = normalization, bgsub = bgsub)

def call_xrd(args):
    """
    Input: args, a value returned by argparse.ArgumentParser.parse_args()

    Calls the xrd sub-command of this script.
    """
    from mpi4py import MPI
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
        normalization = normalization, maxpeaks = maxpeaks, plot_progression = args.plot_progression)
    if not config.testing:
        MPI.Finalize()

def call_histogram(args):
    import summarymetrics
    labels = args.labels
    detid = args.detid
    nbins = args.nbins
    summarymetrics.main(labels, detid, funcstr = args.function, filtered = args.filter, nbins = nbins, separate = args.separate)

def call_datashow(args):
    import datashow
    labels = args.labels
    detid = args.detid
    rmax = args.max
    rmin = args.min
    datashow.main(labels, detid, path = args.output, masked = args.masks, rmin = rmin, rmax = rmax, run = args.run)

def call_eventframes(args):
    import eventframes
    eventframes.main(args.label, args.detid, filtered = args.filter)

def call_query(args):
    import query
    if args.filter_function:
        filter_function = eval('config.' + args.filter_function)
    else:
        filter_function = None
    if args.filter_function:
        if not args.filter_detid:
            raise ValueError("FILTER_DETID must be provided along with FILTER_FUNCTION")
    query.main(args.querylist, filter_function, args.filter_detid)

def call_showderived(args):
    import query
    print '\n'.join(query.get_derived_datset_labels())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noplot', '-n', action = 'store_true', help = 'If selected, plotting is suppressed')
    subparsers = parser.add_subparsers(help='sub-command help', dest = 'command')

    # Add sub-commands to parser
    argument_parsers.addparser_init(subparsers)
    argument_parsers.addparser_xes(subparsers)
    argument_parsers.addparser_xrd(subparsers)
    argument_parsers.addparser_histogram(subparsers)
    argument_parsers.addparser_datashow(subparsers)
    argument_parsers.addparser_eventframes(subparsers)
    argument_parsers.addparser_query(subparsers)
    argument_parsers.addparser_showderived(subparsers)

    args = parser.parse_args()

    def mongo_commit():
        if utils.isroot():
            if cmd in ['spectrum', 'xrd', 'histogram', 'datashow']:
                database.mongo_commit(args.labels)
            elif cmd == 'eventframes':
                database.mongo_commit([args.label])

    config_source = utils.resource_path('data/config.py')
    config_dst = 'config.py'
    cmd = vars(args)['command']

    # try to execute from database
    key = '_'.join(sys.argv[1:])

    database.mongo_init(key)

    if cmd == 'init': # Enter init sub-command
        call_init(config_source, config_dst)
    if args.noplot:
        config.noplot = True

    if config.playback:
        try:
            playback.load_db(key)
            playback.execute()
            mongo_commit()
            sys.exit(0)
        except KeyError:
            pass

    import os

    if cmd != 'init':
        if not os.path.isfile(config_dst):
            parser.error("File config.py not found. Run 'python dataccess.py init' to initialize config.py, and then edit it appropriately before re-running")
        elif cmd == 'spectrum':
            call_xes(args)
        elif cmd == 'xrd':
            call_xrd(args)
        elif cmd == 'histogram':
            call_histogram(args)
        elif cmd == 'datashow':
            call_datashow(args)
        elif cmd == 'eventframes':
            call_eventframes(args)
        elif cmd == 'query':
            call_query(args)
        elif cmd == 'showderived':
            call_showderived(args)

    if utils.isroot():
        if config.playback:
            playback.save_db(key)
            print playback.db
            playback.execute()
        mongo_commit()
    return key

if __name__ == '__main__':
    main()
#comm = MPI.COMM_WORLD
#comm.Barrier()
#MPI.Finalize()

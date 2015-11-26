#!/reg/g/psdm/sw/releases/ana-current/arch/x86_64-rhel7-gcc48-opt/bin/python2.7

import sys
# print sys.path
# print(sys.executable)
#import ipdb
# ipdb.set_trace()

sys.path.insert(
    1,
    '/reg/g/psdm/sw/external/python/2.7.10/x86_64-rhel7-gcc48-opt/lib/python2.7/site-packages')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
sys.path.append(
    '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/pathos-0.2a1.dev0-py2.7.egg')
sys.path.append(
    '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dataccess-1.0-py2.7.egg')
import os
import shutil
import argparse
from dataccess import utils


def addparser_init(subparsers):
    init = subparsers.add_parser(
        'init', help='Initialize config.py in local directory')
    # Unfortunate hack to figure out which subcommand was called. It looks
    # like the dest keyword for add_parser isn't usable in python 2.7.
    init.add_argument('--initcalled', help='dummy argument')


def call_init(config_source, config_dst):
    """
    Inputs:
        config_source, the path of this program's config file template.
        config_dst, path to which to copy the config file template

    Initializes config.py in the working directory
    """
    shutil.copyfile(config_source, config_dst)


def addparser_sync(subparsers):
    # google spreadsheet sync sub-command
    sync = subparsers.add_parser(
        'sync',
        help='Launch background process to synchonize google spreadsheet data. Spreadsheet URL and sheet number(s) must be provided in config.py')
    # TODO: handle loading of url from config.py
    sync.add_argument('--url', help='url of the google spreadsheet')
    sync.add_argument('--sheet', '-s', type=int, default=0,
                      help='Index of the sheet (defaults to 0)')


def call_sync(url, sheet):
    # TODO: complete this function
    print 'calling callsync'
    pass


def addparser_xes(subparsers):
    # XES analysis sub-command
    xes = subparsers.add_parser(
        'xes', help='Process CSPAD 140K data into XES spectra')
    xes.add_argument('detid', type=int, help='Detector ID.')
    xes.add_argument(
        'datapaths',
        nargs='+',
        help='Paths of CSPAD data files to process into spectra and plot.')
    xes.add_argument('--pxwidth', '-p', type=int, default=3,
                     help='Pixel width of CSPAD subregion to sum.')
    xes.add_argument(
        '--calibration',
        '-c',
        type=str,
        help='Path of data to use for calibration of the energy scale (if --energy_ref1_energy_ref2_calibration is selected but a calibration file is not provided). If not provided this parameter defaults to the first path in datapaths.')
    xes.add_argument(
        '--subtraction',
        '-d',
        type=str,
        help='Path of a data file to use as a dark frame subtraction')
    xes.add_argument(
        '--energy_ref1_energy_ref2_calibration',
        '-k',
        action='store_true',
        help='Enable automatic generation of energy calibration based on k alpha and k beta peak locations if --calibration_load_path is not given.')
    xes.add_argument(
        '--eltname',
        '-e',
        default='',
        help='Element name. This parameter is required for generating an energy scale using --energy_ref1_energy_ref2_calibration.')
    xes.add_argument(
        '--calibration_save_path',
        '-s',
        type=str,
        help='Path to which to save energy calibration data if calibration_load_path is unspecified and --energy_ref1_energy_ref2_calibration is selected.')
    xes.add_argument('--calibration_load_path', '-l', type=str,
                     help='Path from which to load energy calibration data.')
    xes.add_argument(
        '--runlabels',
        '-r',
        type=str,
        default='labels.txt',
        help='Path to run group label input file. Defaults to labels.txt. If the file does not exist it is generated automatically (based on run data timestamps), and can subsequently be edited by the user.)')


def call_xes(args):
    """
    Input: args, a value returned by argparse.ArgumentParser.parse_args()

    Calls the xes sub-command of this script.
    """
    paths = args.datapaths
    eltname = args.eltname
    if args.calibration:
        calibration = args.calibration
    else:
        calibration = paths[0]
    cold_calibration_path = calibration
    pxwidth = args.pxwidth
    pxwidth = args.pxwidth
    calibration_save_path = args.calibration_save_path
    calibration_load_path = args.calibration_load_path
    dark_path = args.subtraction
    energy_ref1_energy_ref2_calibration = args.energy_ref1_energy_ref2_calibration
    xes_process.main(
        args.detid,
        paths,
        cold_calibration_path=cold_calibration_path,
        pxwidth=pxwidth,
        calib_save_path=calibration_save_path,
        calib_load_path=calibration_load_path,
        dark_path=dark_path,
        energy_ref1_energy_ref2_calibration=energy_ref1_energy_ref2_calibration,
        eltname=eltname,
        run_label_filename=args.runlabels)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

# Add sub-commands to parser
addparser_init(subparsers)
addparser_sync(subparsers)
addparser_xes(subparsers)

args = parser.parse_args()

config_source = utils.resource_path('data/config.py')
config_dst = 'config.py'
if 'initcalled' in args:  # Enter init sub-command
    call_init(config_source, config_dst)
else:
    if not os.path.isfile(config_dst):
        parser.error(
            "File config.py not found. Run 'python dataccess.py init' to initialize config.py, and then edit it appropriately before re-running")
    from dataccess import xes_process
    if 'url' in args:  # Enter google spreadsheet sync sub-command
        call_sync(args.url, args.sheet)

    elif 'detid' in args:  # Enter xes sub-command
        call_xes(args)

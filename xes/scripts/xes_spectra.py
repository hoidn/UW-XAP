#!/usr/bin/env python
import argparse
from xes import xes_process

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapaths', nargs = '+', help = 'Paths of CSPAD data files to process into spectra and plot.')
    parser.add_argument('--pxwidth', '-p', type = int, default = 3, help = 'Pixel width of CSPAD subregion to sum.')
    parser.add_argument('--calibration', '-c', type = str, help = 'Path of data to use for calibration of the energy scale (if --kalpha_kbeta_calibration is selected but a calibration file is not provided). If not provided this parameter defaults to the first path in datapaths.')
    parser.add_argument('--subtraction', '-d', type = str, help = 'Path of a data file to use as a dark frame subtraction')
    parser.add_argument('--kalpha_kbeta_calibration', '-k', action = 'store_true', help = 'Enable automaic generation of energy calibration based on k alpha and k beta peak locations if --calibration_load_path is not given.')
    parser.add_argument('--eltname', '-e', default = '', help = 'Element name. This parameter is required for generating an energy scale using --kalpha_kbeta_calibration.')
    parser.add_argument('--calibration_save_path', '-s', type = str, help = 'Path to which to save energy calibration data if calibration_load_path is unspecified and --kalpha_kbeta_calibration is selected.')
    parser.add_argument('--calibration_load_path', '-l', type = str, help = 'Path from which to load energy calibration data.')
    args = parser.parse_args()
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
    kalpha_kbeta_calibration = args.kalpha_kbeta_calibration
    xes_process.main(paths, cold_calibration_path = cold_calibration_path,
        pxwidth = pxwidth,
        calib_save_path = calibration_save_path, calib_load_path =
            calibration_load_path, dark_path = dark_path,
            kalpha_kbeta_calibration = kalpha_kbeta_calibration,
            eltname = eltname)

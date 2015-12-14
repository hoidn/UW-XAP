import os
import time
import sys
import argparse
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/pathos-0.2a1.dev0-py2.7.egg')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dataccess-1.0-py2.7.egg')
from dataccess import data_access
from dataccess import avg_bgsubtract_hdf
import config


d4 = avg_bgsubtract_hdf.get_signal_bg_one_run(688, mode = 'script')

def generate_all_batches(search_range = (-float('inf'), float('inf'))):
    rangemin, rangemax = search_range
    all_runs = data_access.get_all_runs()
    commands = []
    for run in all_runs:
        if run >= rangemin and run <= rangemax:
            for detid in config.detID_list:
                commands.append(avg_bgsubtract_hdf.get_signal_bg_one_run(run, detid = detid, mode = 'script'))
    return commands

def submit_all_batches(search_range = (-float('inf'), float('inf'))):
    commands = generate_all_batches(search_range = search_range)
    for command in commands:
        os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min', type = int)
    parser.add_argument('--max', type = int)
    parser.add_argument('--generate', '-g', action = 'store_true')
    parser.add_argument('--submit', '-s', action = 'store_true')
    args = parser.parse_args()

    if args.min:
        amin = args.min
    else:
        amin = -float('inf')
    if args.max:
        amax = args.max
    else:
        amax = float('inf')
    search_range = (amin, amax)

    if args.generate:
        generate_all_batches(search_range = search_range)
    if args.submit:
       submit_all_batches(search_range = search_range)

    time.sleep(1000)

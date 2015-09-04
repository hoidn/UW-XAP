#!/usr/bin/env python

import time
import sys
import json
import webbrowser
import logging
import argparse

import httplib2
import os.path


from apiclient import discovery
from oauth2client import client
from oauth2client.file import Storage

import gspread
import pandas as pd
import numpy as np
import pdb
from atomicfile import AtomicFile

from dataccess import utils

RUN_HEADER = 'Run #'
LABELS = ['label 1', 'label 2']

# TODO: better validation of user input

def acquire_oauth2_credentials(secrets_file):
    """Flows through OAuth 2.0 authorization process for credentials."""
    flow = client.flow_from_clientsecrets(
        secrets_file,
        scope='https://spreadsheets.google.com/feeds',
        redirect_uri='urn:ietf:wg:oauth:2.0:oob')
    
    auth_uri = flow.step1_get_authorize_url()
    webbrowser.open(auth_uri)
    
    auth_code = raw_input('Enter the authentication code: ')
    
    credentials = flow.step2_exchange(auth_code)
    return credentials

def get_run_ranges(url, sheet_number = 0):
    """
    Return 2d list of the format

    [[start run, end run, label1, label2], 
    ......]
    corresponding to the contents of the spreadsheet at the url
    """
    def to_ndarr(list_2d):
        return np.array([np.array(row) for row in list_2d])

    def merge_horizontal(arr1, arr2):
        return [r1 + r2 for r1, r2 in zip(arr1, arr2)]
            

    def fill_with_None(list_2d):
        maxlen = max(map(len, list_2d))
        newl = []
        for row in list_2d:
            newl.append(row + [None] * (maxlen - len(row)))
        return newl

    def fill_to_match(arr1, arr2, target_num_labels = 2):
        if len(arr1) > len(arr2):
            return arr1, arr2 + [[None] * target_num_labels] * (len(arr1) - len(arr2))
        else:
            return arr1 + [target_num_labels * [None]] * (len(arr2) - len(arr1)), arr2
    def str_to_range(run_string):
        """
        Converts string of format "abcd,efgh" to list [abcd, efgh]

        Returns ValueError on incorrectly-formatted range entries.
        """
        if not run_string:
            return [None, None]
        else:
            split = run_string.split('-')
            try:
                map(int, split) # values convertible to ints?
                if len(split) == 1:
                    return 2 * split
                elif len(split) == 2:
                    return split
                else:
                    raise ValueError("Invalid run range format: ", run_string)
            except:
                raise ValueError("Invalid run range format: ", run_string)

    storage =  Storage(utils.resource_path('data/credentials'))
    credentials = storage.get()
    gc = gspread.authorize(credentials)
    document = gc.open_by_url(url)
    worksheet = document.get_worksheet(sheet_number) 
    cols = [worksheet.find(lab).col for lab in LABELS]
    run_col = worksheet.find(RUN_HEADER).col
    runs = map(str_to_range, worksheet.col_values(run_col)[1:])
    labels = fill_with_None([worksheet.col_values(col)[1:] for col in cols])
    
    labels = map(list, zip(*fill_with_None(labels)))
    runs, labels = fill_to_match(runs, labels)
    combined = merge_horizontal(runs, labels)
    return filter(lambda x: x[0], combined)


def write_labels(url, sheet_number = 0):
    dat = get_run_ranges(url, sheet_number = sheet_number)
    arr = np.array([tuple(row) for row in dat], dtype = np.dtype("d, d, U200, U200"))
    with AtomicFile('labels.txt', 'w') as f:
        np.savetxt(f, arr, fmt = ['%04d', '%04d', '%s', '%s'], header = 'start run, end run, label1, label2', delimiter = ',')
    print "wrote to labels.txt"
    return arr

def main(url, sheet_number = 0):
    while 1:
        write_labels(url, )
        time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help = 'url of the google spreadsheet')
    parser.add_argument('--sheet', '-s', type = int, default = 0, help = 'Index of the sheet (defaults to 0)')

    args = parser.parse_args()
    main(args.url, args.sheet)

import argparse
import sys
import os

sys.path.append('.') # so that config.py can be imported
from dataccess import logbook
import config

'Synchronize google spreadsheet logbook data by running this script on pslogin03'

parser = argparse.ArgumentParser()
parser.add_argument('--url', help = 'URL of the google spreadsheet. Defaults to the URL provided in config.py.')
args = parser.parse_args()

if args.url:
    url = args.url
else:
    url = config.url
logbook.main(url)

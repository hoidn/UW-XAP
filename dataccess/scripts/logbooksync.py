import argparse
import sys
import os

sys.path.append('.') # so that config.py can be imported
from dataccess import logbook

'Synchronize google spreadsheet logbook data by running this script on pslogin'

parser = argparse.ArgumentParser()
parser.add_argument('--url', help = 'URLs of the google drive spreadsheet. Defaults to the URL provided in config.py.')
args = parser.parse_args()

if args.url:
    url = args.url
else:
    try:
        import config
        url = config.url
    except ImportError:
        raise ImportError("config.py not found. config.py is necessary to load logbook URL if the optional argument --url is not provided.")
logbook.main(url)

import argparse
import sys
import os

sys.path.append('.') # so that config.py can be imported
from dataccess import logbook

'Synchronize google spreadsheet logbook data by running this script on pslogin03'

parser = argparse.ArgumentParser()
parser.add_argument('--urls', nargs = '+', help = 'URLs of the google drive spreadsheets. Defaults to the URLs provided in config.py.')
args = parser.parse_args()

if args.urls:
    urls = args.urls
else:
    try:
        import config
        urls = config.urls
    except ImportError:
        raise ImportError("config.py not found. config.py is necessary to load logbook URL if the optional argument --urls is not provided.")
logbook.main(urls)

#from distutils.core import setup

import sys

from setuptools import setup, find_packages

setup(name='dataccess',
    version='1.0',
    packages = find_packages('.'),
    package_dir={'dataccess': 'dataccess'},
    package_data={'dataccess': ['data/*']},
    scripts = [
        'scripts/mecana.py', 'scripts/g_spreadsheet_sync.py'
    ],
    zip_safe = False,
    )

print  "Packages are: ", find_packages('.')

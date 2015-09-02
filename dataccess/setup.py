#from distutils.core import setup

from setuptools import setup, find_packages

setup(name='dataccess',
    version='1.0',
    packages = find_packages('.'),
    package_dir={'dataccess': 'dataccess'},
    package_data={'dataccess': ['data/*']},
    scripts = [
        'scripts/XES_converter.py', 'scripts/g_spreadsheet_sync.py'
    ],
    #zip_safe = False,
    )

print  find_packages('.')

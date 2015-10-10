from setuptools import find_packages, setup
#from distutils.core import setup

setup(name='xes',
    version='1.0',
    packages = find_packages('.'),
    package_dir={'xes': 'xes'},
    package_data={'xes': ['data/*']},
    scripts = [
        'scripts/xes_spectra.py'
    ],
    zip_safe = False,
    )

print  find_packages('.')

#from distutils.core import setup

import sys
import os
import pip

from setuptools import setup, find_packages

pip.main(['install',  'pyzmq', '--install-option=--zmq=bundled'])

setup(name='dataccess',
    version='1.0',
    packages = find_packages('.'),
    package_dir={'dataccess': 'dataccess'},
    package_data={'dataccess': ['data/*']},
    scripts = [
        'scripts/mecana.py', 'scripts/logbooksync.py'
    ],
    install_requires = ['recordclass', 'google-api-python-client', 'httplib2', 'atomicfile', 'urllib3', 'gspread', 'requests>=2.9.1', 'multiprocess', 'dill', 'pox', 'ppft', 'ipdb', 'joblib'],
    zip_safe = False,
    )

pip.main(['install', 'git+https://github.com/uqfoundation/pathos.git'])
pip.main(['install', 'pymongo'])
pip.main(['install', 'pytest'])

print  "Packages are: ", find_packages('.')

# fix hash-bang line for mecana
#home = os.environ['HOME']
#pypath = '/reg/g/psdm/sw/releases/ana-current/arch/x86_64-rhel7-gcc48-opt/bin/python'
#with open(home + '/.local/bin/mecana.py', 'r') as f:
#    mecana = f.read()
#with open(home + '/.local/bin/mecana.py', 'w') as f:
#    f.write('#!' + pypath + '\n' + mecana)

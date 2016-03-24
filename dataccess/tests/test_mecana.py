from dataccess import mecana_main
from pymongo import MongoClient
from dataccess import database
import sys
import ipdb
import os

def reset():
    os.system('rm -rf cache')
    os.system('rm -rf db')
    database.delete_collections()
    database.collections_lookup['session_cache'].delete_many({})

def test_xrd():
    reset()
    os.system('mecana.py -n xrd quad2 evaltest -b -c Fe3O4')
    reset()

def test_xrd_2():
    """
    Make a query and then run mecana.py xrd on the resulting dataset.
    """
    reset()
    os.system('mecana.py -n query runs 530 535')
    reset()
    os.system('mecana.py -n xrd quad2 runs-530.0-535.0 -b -c Fe3O4')
    reset()

def test_datashow():
    sys.argv = ['mecana.py'] + '-n datashow quad1 530'.split()
    return mecana_main.main()


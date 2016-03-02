from dataccess import mecana_main
from pymongo import MongoClient
from dataccess import database
import sys
import ipdb
import os

#def test_xrd():
#    os.system('rm -rf cache')
#    os.system('rm -rf db')
#    database.delete_collections()
#    sys.argv = ['mecana.py'] + '-n xrd quad2 evaltest -b -c Fe3O4'.split()
#    print 'SYS ARGV IS', sys.argv
#    key = mecana_main.main()
#    MONGO_PORT = database.MONGO_PORT
#    MONGO_HOST = database.MONGO_HOST
#    result = list(database.collections_lookup['session_cache'].find({'key': key}))[0]
#    assert 'data' in result
#    return result['data']

def test_datashow():
    sys.argv = ['mecana.py'] + 'mecana.py datashow quad1 64'.split()
    key = mecana_main.main()

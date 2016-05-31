import sys
#sys.path.insert(0, '/reg/neh/home/ohoidn/anaconda2/lib/python2.7/')
#del sys.modules['pickle']

from dataccess import data_access as data
from dataccess import utils
from dataccess import query
import pdb
import time

def even_filter(arr, nevent = None, **kwargs):
    return bool(nevent % 2)

def test_get_label_data():
    from dataccess import database
    database.delete_collections()
    d = query.DataSet.from_query(query.query_list([('label', r"evaltest"), ('transmission', 0.1)]))
    time.sleep(4)
    si = data.eval_dataset('label-evaltest-transmission-0.1', 'si', event_data_getter = utils.usum)
    assert si[1][620][30]
    
    d_filtered = query.DataSet.from_query(query.query_list([('label', r"evaltest"), ('transmission', 0.1)]), event_filter = even_filter, event_filter_detid = 'si')
    unfiltered = data.eval_dataset(d.label, 'si')
    filtered = data.eval_dataset(d_filtered.label, 'si')
    return unfiltered, filtered

#def test_get_dataset_attribute_value():
#    query.existing_dataset_by_label('label-evaltest-transmission-0.1').get_attribute('transmission') ==\
#        0.1
## TODO: fix regex handling (i.e. the below test case again)
#    query.existing_dataset_by_label('label-e.*est-transmission-0.1').get_attribute('transmission') ==\
#        0.1
#    query.existing_dataset_by_label('evaltest').get_attribute('transmission') ==\
#        0.1
#    query.existing_dataset_by_label('fe3o4lab1').get_attribute('runs') ==\
#        (608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 607)
#    query.existing_dataset_by_label('620').get_attribute('runs') ==\
#        (620,)
## TODO: fix this as well
#    query.existing_dataset_by_label('205-206').get_attribute('runs') ==\
#        (205, 206)

def test_get_dark_data():
    time.sleep(4)
    assert data.get_dark_dataset('evaltest').label == '377'
    assert data.get_dark_dataset('fe3o4lab1').label== '436'

def test_get_data_and_filter_logbook():
    assert data.eval_dataset('40', 'quad1')

def test_add_datasets():
    r1 = data.get_dark_dataset('evaltest').evaluate('si')
    r2 = data.get_dark_dataset('fe3o4lab1').evaluate('si')
    r3 = r1 + r2
    assert r3.nevents()  == r1.nevents() + r2.nevents()

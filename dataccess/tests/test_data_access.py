import numpy as np
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

def test_dark():
    dark = data.get_dark_dataset('fe3o4lab1').evaluate('si')
    raw = data.eval_dataset('fe3o4lab1', 'si')
    subtracted = data.eval_dataset_and_filter('fe3o4lab1', 'si', darksub = True)

#difference = subtracted.mean - (raw.mean - dark.mean)
#
#raw.event_data.keys()
#
#dark.event_data.keys()
#
#from dataccess import query
#
#ds = query.existing_dataset_by_label('436')
#
##dark = data.get_dark_dataset('r206').evaluate('si')
#
##dark = data.get_dark_dataset('fe3o4lab1').evaluate('si')
#
#raw = data.eval_dataset_and_filter('r206', 'si', darksub = False)
#
#darkds = query.DataSet([200], label = 'r200')
#dark = data.eval_dataset_and_filter(darkds, 'si', darksub = False)
#
#raw2 = data.eval_dataset('r206', 'si', dark_frame = dark.mean)
#
#dark3 = np.zeros_like(raw2.mean)
#raw3 = data.eval_dataset('r206', 'si', dark_frame = dark3)
#
#d = np.ones_like(dark.mean) + dark.mean
#null = data.eval_dataset(darkds, 'si', dark_frame = d)
#
## orig: 40041.109465060421
#np.mean(dark.mean)
#
#d = data.eval_dataset(data.get_dark_dataset('r436'), 'si')
## orig: 7.7041766086471402
#np.mean(d.mean)
#
#dark.event_data.keys()
#
#dark.mean
#
#r206 = query.DataSet([206], label = 'r206').evaluate('si')
#
#
#r4 = query.DataSet([436], label = 'r436')
#
#ds.__dict__
#
#d2 = r4.evaluate('si')
#
#data.get_dark_dataset('r436')
#

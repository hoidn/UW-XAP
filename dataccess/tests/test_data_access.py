import sys
#sys.path.insert(0, '/reg/neh/home/ohoidn/anaconda2/lib/python2.7/')
#del sys.modules['pickle']

from dataccess import data_access as data
from dataccess import query
import ipdb

def even_filter(arr, nevent = None, **kwargs):
    return bool(nevent % 2)

def test_get_dataset_attribute_value():
    assert data.get_dataset_attribute_value('label-evaltest-transmission-0.1', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('label-e.*est-transmission-0.1', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('evaltest', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('fe3o4lab', 'runs') == (608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 602, 603, 604, 605, 606, 607)
    assert data.get_dataset_attribute_value('205', 'runs') == (205,)
    assert data.get_dataset_attribute_value('205-206', 'runs') == (205, 206)

def test_get_label_data():
    from dataccess import database
    database.delete_collections()
    d = query.DataSet(query.query_list([('label', r"evaltest"), ('transmission', 0.1)]))
    si = data.get_label_data('label-evaltest-transmission-0.1', 'si')
    assert si[1][620][30]
    
    d_filtered = query.DataSet(query.query_list([('label', r"evaltest"), ('transmission', 0.1)]), event_filter = even_filter, event_filter_detid = 'si')
    unfiltered = data.get_data_and_filter(d.label, 'si')
    filtered = data.get_data_and_filter(d_filtered.label, 'si')
    return unfiltered, filtered

def test_get_dark_data():
    assert data.get_dark_label('evaltest') == '377'
    assert data.get_dark_label('fe3o4lab') == '436'

def test_get_data_and_filter_logbook():
    assert data.get_data_and_filter_logbook('40', 'quad1')

from dataccess import data_access as data
from dataccess import query
import ipdb

def even_filter(arr, nevent = None, **kwargs):
    return bool(nevent % 2)

def test_get_dataset_attribute_value():
    assert data.get_dataset_attribute_value('label-evaltest-transmission-0.1', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('label-e.*est-transmission-0.1', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('evaltest', 'transmission') == 0.1

def test_get_label_data():
    d = query.DataSet(query.query_list([('label', r"evaltest"), ('transmission', 0.1)]))
    si = data.get_label_data('label-evaltest-transmission-0.1', 'si')
    assert si[1][620][304] ==  33213729
    
    d_filtered = query.DataSet(query.query_list([('label', r"evaltest"), ('transmission', 0.1)]), event_filter = even_filter, event_filter_detid = 'si')
    unfiltered = data.get_data_and_filter('label-evaltest-transmission-0.1', 'si')
    filtered = data.get_data_and_filter('label-evaltest-transmission-0.1-filter-even_filter-8178cd20898725203c99b4008d93a8e0fc40f70c-si', 'si')
    return unfiltered, filtered

unfiltered, filtered = test_get_label_data()


from dataccess import data_access as data

def test_get_dataset_attribute_value():
    assert data.get_dataset_attribute_value('label-evaltest-transmission-0.1', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('label-e.*est-transmission-0.1', 'transmission') == 0.1
    assert data.get_dataset_attribute_value('evaltest', 'transmission') == 0.1

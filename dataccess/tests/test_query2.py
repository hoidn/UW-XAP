import config
from dataccess import query
from dataccess import database
import numpy as np
import time
import pytest
import time

def test_dataset():
    database.delete_collections(delete_logbook = True)
    time.sleep(4)
    tdict = {'runs': (620, 621)}
    ds = query.DataSet.from_logbook_label_dict(tdict, 'transmission')
    assert ds.runs == (620, 621)

    assert ds.get_attribute('transmission') == 0.1

    with pytest.raises(Exception) as e:
        tdict2 = {'runs': (None)}
        ds2 = query.DataSet.from_logbook_label_dict(tdict2, 'tl2')

def test_conflicting_labels():
    database.delete_collections(delete_logbook = True)
    time.sleep(4)
    def tfilter(nevent = None, run = None, **kwargs):
        return bool(nevent % 2)

    td = query.main('material dark'.split(), label = 'tlabel')
    with pytest.raises(Exception) as e:
        td2 = query.main('material dark'.split(), label = 'tlabel', event_filter = tfilter,
            event_filter_detid = 'si' )
    td2 = query.main('material dark'.split(), label = 'tlabel2', event_filter = tfilter,
        event_filter_detid = 'si' )
    td3 = query.main('material dark'.split(), label = 'tlabel3')
    td4 = query.main('material dark'.split(), label = 'tlabel')

    assert td == td3 == td4
    assert td != td2

import config
from dataccess import query
import numpy as np
import time
import pytest

def test_dataset():
    tdict = {'runs': (620, 621)}
    ds = query.DataSet.from_logbook_label_dict(tdict, 'transmission')
    assert ds.runs == (620, 621)

    assert ds.get_attribute('transmission') == 0.1

    with pytest.raises(Exception) as e:
        tdict2 = {'runs': (None)}
        ds2 = query.DataSet.from_logbook_label_dict(tdict2, 'tl2')

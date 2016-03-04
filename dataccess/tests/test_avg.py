from dataccess import avg_bgsubtract_hdf
import numpy as np
import ipdb

def test_get_signal_many_parallel():
    #ipdb.set_trace()
    assert not np.all(avg_bgsubtract_hdf.get_signal_many_parallel([620, 621], 'si')[0] == 0)
    return avg_bgsubtract_hdf.get_signal_many_parallel([620, 621], 'si')

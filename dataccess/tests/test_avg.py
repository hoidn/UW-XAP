from dataccess import psget
import numpy as np
import ipdb

def test_get_signal_many_parallel():
    #ipdb.set_trace()
    assert not np.all(psget.get_signal_many_parallel([620, 621], 'si')[0] == 0)
    return psget.get_signal_many_parallel([620, 621], 'si')

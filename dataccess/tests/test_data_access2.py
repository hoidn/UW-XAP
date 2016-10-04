import pdb

from dataccess import data_access
import config
import numpy as np

config.testing = False

def test_filter():
    def data_getter(imarr, **kwargs):
        return np.sum(imarr)

    def filter1(imarr, nevent = None, **kwargs):
        return nevent % 2

    def filter2(imarr, nevent = None, **kwargs):
        return (not nevent % 2)

    # TODO: update this
    ds1 = data_access.eval_dataset_and_filter('206', 'si', event_data_getter = data_getter)
    ds2 = data_access.eval_dataset_and_filter('206', 'si',  event_data_getter = data_getter)

    ds3 = data_access.eval_dataset_and_filter('206', 'si', event_data_getter = data_getter)
    assert ds1.nevents() + ds2.nevents() == ds3.nevents()
    assert np.all(np.isclose(ds1.nevents() * ds1.mean + ds2.nevents() * ds2.mean,
        ds3.nevents() * ds3.mean))
            

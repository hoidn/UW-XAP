import numpy as np
from dataccess import utils

def test_merge_lists():
    a = [[np.ones(10), np.ones(10)]]
    b = [[np.zeros(10), np.zeros(10)]]
    target =\
        [[np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
        np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,  0.,  0.,  0.])]]
    result = utils.merge_lists(a, b)
    assert np.all(np.isclose(target, result))

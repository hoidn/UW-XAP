import numpy as np
from dataccess import utils
from dataccess import utils
import ipdb
from dataccess.output import rprint

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

def test_stdout_to_file():
    @utils.stdout_to_file(path = None)
    def hello():
        rprint( 'hello')
    @utils.stdout_to_file(path = 'tpath')
    def world():
        rprint( 'world')
    def foo():
        rprint( 'foo')
    hello()
    world()
    foo()

    with open('tpath', 'r') as f:
        assert f.read() == 'world\n'

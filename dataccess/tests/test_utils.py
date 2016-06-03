import numpy as np
from dataccess import utils
from dataccess import output
import ipdb
from dataccess.output import log
import os

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
    if os.path.exists('tpath'):
        os.system('rm tpath')
    @output.stdout_to_file(path = None)
    def hello():
        log( 'hello')
    @output.stdout_to_file(path = 'tpath')
    def world():
        log( 'world')
    def foo():
        log( 'foo')
    hello()
    world()
    foo()

    with open('tpath', 'r') as f:
        assert f.read() == 'world\n'

def test_hash_obj():
    def func(x):
        return x**2
    h1 = utils.hash_obj(func)
    def func(x):
        return x**3
    h2 = utils.hash_obj(func)
    def func(x):
        return x**2
    h3 = utils.hash_obj(func)
    assert h1 != h2
    assert h1 == h3
    utils.hash_obj([h1, h2])
    utils.hash_obj([])

    d1 = {206: {0: True, 1: False, 2: True, 3: False}}
    d2 = {206: {0: 0, 1: 1, 2: 0, 3: 1}}
    h4 = utils.hash_obj(d1)
    h5 = utils.hash_obj(d2)
    assert h4 != h5

    h6 = utils.hash_obj(np.ones((10, 10)))

    def func2(x):
        def inner(y):
            return x + y
        return inner
    f = func2(4)
    h7 = utils.hash_obj(f)

def test_all_isinstance():
    assert utils.all_isinstance(range(10), int)

def test_dicts_take_intersecting_keys():
    assert utils.dicts_take_intersecting_keys({}, {}) == ({}, {})
    assert utils.dicts_take_intersecting_keys({1: {2: 3}}, {1: {2: 6, 4: 5}}) ==\
            ({1: {2: 3}}, {1: {2: 6}})

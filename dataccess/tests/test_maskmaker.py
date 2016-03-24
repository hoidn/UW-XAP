from dataccess import maskmaker
import numpy as np

def test_makemask():
    image = np.array([[0., 0., 0., 0., 1., 1., 1., 0], [0., 0., 0., 0., 1., 1., 1.,  0]])
    target = np.array([[False, False, False, False, False,  True, False, False],
       [False, False, False, False, False,  True, False, False]], dtype=bool)
    assert np.all(maskmaker.makemask(image, 1) == target)

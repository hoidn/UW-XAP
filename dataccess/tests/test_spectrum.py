from dataccess import xes_process as s
from utils import *
import os
#
#os.system('rm -rf spectra')
#os.system('touch spectra')

def test_get_plotinfo_one_label():
    import dill
    with open('xes_process_get_plotinfo_one_label_1.p', 'r') as f:
        target = dill.load(f)
    result = s.get_plotinfo_one_label('si', '400', s.get_spectrum, event_indices=[0, 10])
    assert npcomp(target, result)

    with open('xes_process_get_plotinfo_one_label_2.p', 'r') as f:
        target = dill.load(f)
    result = s.get_plotinfo_one_label('si', 'evaltest', s.get_spectrum)
    assert npcomp(target, result)

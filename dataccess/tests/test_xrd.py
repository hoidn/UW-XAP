#import pytest
from dataccess import xrd
import numpy as np
from dataccess import utils
from dataccess import query
from dataccess import geometry
import config

config.noplot = True

# TODO: repopulate this module

#def test_main():
#    result = xrd.main(['quad2'], ['evaltest'], bgsub = False)
#    assert utils.make_hashable(result)  == '810b4ce967076eb29c140561c5c2195812fc6657'
#    patterns, imarrays = result
#    assert np.all(np.isclose(patterns[0][1][200:300],
#        np.array([ 603.41339553,  604.60241147,  604.8132577 ,  605.00970532,
#                605.33371694,  605.6524121 ,  606.25001623,  606.4694361 ,
#                607.09366648,  607.46448784,  607.82977002,  608.04411829,
#                608.094091  ,  608.45238118,  608.55971817,  608.98025408,
#                609.18042089,  609.33096877,  609.64706146,  609.23083376,
#                609.7201641 ,  609.67828102,  609.50995183,  609.19411749,
#                608.95988306,  609.12589577,  608.82251798,  609.13108028,
#                608.87665294,  609.08883371,  609.41371644,  609.43804357,
#                609.53530761,  609.27447619,  609.49108276,  609.38727837,
#                609.64542097,  609.78953353,  609.681366  ,  609.60355354,
#                609.11720897,  609.35014272,  609.49232533,  609.04695572,
#                609.73198243,  609.4651523 ,  609.74775911,  609.6149503 ,
#                609.52848467,  609.73824674,  609.5324574 ,  609.65942115,
#                609.72913484,  609.60777801,  609.98749944,  609.58375773,
#                610.00112405,  609.94952124,  609.87619306,  610.02360954,
#                609.64504285,  610.06889023,  609.97637508,  610.15339723,
#                610.3380583 ,  609.87963519,  610.16181761,  609.89960421,
#                610.04445114,  610.18748158,  609.60911524,  609.88626533,
#                609.91440924,  609.93890382,  610.04859453,  609.89445539,
#                609.74131792,  609.74048015,  609.82546076,  610.01093221,
#                609.89626933,  609.94661208,  609.62426225,  609.62100503,
#                609.44558921,  609.26088109,  609.59175626,  609.6372742 ,
#                609.94691263,  609.97322182,  610.08944111,  609.74603612,
#                609.49712074,  609.84825641,  609.81642711,  609.974684  ,
#                609.70636956,  609.76026202,  609.59174468,  609.61337859])))



# TODO: what's the reason for this regression?
#def test_mecana_xrd():
#    import test_mecana
#    test_mecana.reset()
#    os.system('mecana.py -n xrd quad2 -l evaltest -b -c Fe3O4')
#
#    from dataccess import mecana_main
#    import sys
#    sys.argv = 'mecana.py -n xrd quad2 -l evaltest -b -c Fe3O4'.split()
#    if config.playback:
#        with pytest.raises(SystemExit):
#            mecana_main.main()
#    else:
#        mecana_main.main()

def test_xrd_process_imarray():
    arr = np.zeros((819, 819))
    geometry.process_imarray('quad2', arr, compound_list = ['MgO'], pre_integration_smoothing = 0.)


#def test_peak_sizes():
#    def my_200_array(arr, bgsub = False, **kwargs):
#        dss = xrd.Dataset(arr, 'array', ['quad2'], ['MgO'])
#        angles, intensities, _ = xrd.process_dataset(dss, bgsub = bgsub)
#        return xrd.peak_sizes(angles, intensities, dss.compound_list[0])
#    tarr = np.ones((819, 819))
#    assert np.all(np.isclose(my_200_array(tarr),  [  0.,          0.,         66.7233081]))


def my_200_array(arr, bgsub = False, **kwargs):
    dss = xrd.Pattern.from_dataset(arr, 'quad2', ['MgO'], label  = 'test')
    return dss.peak_sizes()

def get_query_dataset(querystring, alias = None, print_enable = True, **kwargs):
    dataset = query.main(querystring.split(), label = alias, **kwargs)
    if utils.isroot() and print_enable:
        print "Runs: ", dataset.runs
    return dataset

def test_peak_sizes():
    d1 = get_query_dataset('runs 906 906')
    arr = d1.evaluate('quad2').mean
    assert np.isclose(np.mean(arr), -68.242721825892659)
    peakarr = my_200_array(arr)
    assert np.isclose(peakarr, np.array([  1.10760626,  24.60676483,   0.85673761])).all()

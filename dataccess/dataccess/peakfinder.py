import pdb
import numpy as np

peak_attrs = ['seg', 'row', 'col', 'npix', 'amp_max', 'amp_total', 'row_cgrav', 'col_cgrav', 'raw_sigma', 'col_sigma', 'row_min', 'row_max', 'col_min', 'col_max', 'bkgd', 'noise', 'son']

def make_peak_dict(lst):
    return {k: v for k, v in zip(peak_attrs, lst)}

def consolidate_peaks(nda, winds = None, mask = None, thr_low = 10, thr_high = 150, radius = 5, dr = 1.):
    #pdb.set_trace()
    output = np.zeros_like(nda)
    def add_peak(pk):
        """
        Add the total value of a peak to its center of mass position in the
        output array. 
        """
        value = pk['amp_total']# - pk['npix'] * pk['bkgd']
        i, j = pk['row_cgrav'], pk['col_cgrav']
        def peak_valid():
            n, m = np.shape(nda)
            return (n > i >= 0 and m > j >= 0)
        if peak_valid():
            output[i][j] = value

    from ImgAlgos.PyAlgos import PyAlgos
    if mask is not None:
        assert np.shape(mask) == np.shape(nda)
        
    # create object:
    alg = PyAlgos(windows=winds, mask=mask, pbits=0)
    peaks = map(make_peak_dict, alg.peak_finder_v1(nda, thr_low=thr_low, thr_high=thr_high, radius=radius, dr=dr))

    map(add_peak, peaks)
    return output

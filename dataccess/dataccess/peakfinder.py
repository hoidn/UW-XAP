import pdb
from dataccess import utils
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

@utils.memoize(timeout = None)
def find_islands(detid, threshold, minsize = 75):
    """
    Remove 'islands' meeting the specified minimum size, and consisting of
    pixels with value above the given threshold, by setting them to 0.

    Returns the modified array and a list of ndarray of coordinate pairs for
    the affected pixels for each cluster/island:
        arr -> np.ndarray, clusters -> list of np.ndarray
    """
    import config
    extra_masks = config.detinfo_map[detid].extra_masks
    arr = utils.combine_masks(None, extra_masks, transpose = False)
    q = []
    def search(i, j):
        cluster = []
        q.append((i, j))
        visited[(i, j)] = True
        while q:
            if arr[(i + 1, j)] > threshold and not visited[(i + 1, j)]:
                q.append((i + 1, j))
                visited[(i + 1, j)] = True
            if arr[(i, j + 1)] > threshold and not visited[(i, j + 1)]:
                q.append((i, j + 1))
                visited[(i, j + 1)] = True
            if arr[(i - 1, j)] > threshold and not visited[(i - 1, j)]:
                q.append((i - 1, j))
                visited[(i - 1 , j)] = True
            if arr[(i, j - 1)] > threshold and not visited[(i, j -1 )]:
                q.append((i, j - 1))
                visited[(i , j - 1)] = True
            current = q.pop()
            i, j = current
            cluster.append(current)
        return cluster
    def zero_cluster(cluster):
        for coords in cluster:
            arr[coords] = 0
    visited = np.zeros(arr.shape, dtype = np.bool)
    clusters = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (not visited[i][j]) and (arr[i][j]) > 0:
                cluster = search(i, j)
                if len(cluster) < minsize:
                    zero_cluster(cluster)
                else:
                    clusters.append(cluster)
    return arr, map(np.array, clusters)

def bounding_view(array, cluster):
    def get_bounds(cluster):
        i, j = cluster.T
        return np.min(i), np.max(i), np.min(j), np.max(j)
    imin, imax, jmin, jmax = get_bounds(cluster)
    return array[imin:imax + 1, jmin:jmax + 1]

def clustermap(arr, clusters, func):
    def do_one(cluster):
        return bounding_view(arr, cluster).copy()
    return map(func, map(do_one, clusters))


def peakfilter_frame(arr, detid = None, window_min = 0, radius = 4, thr_low = 20, thr_high = 50, detid_match = lambda detid: True, box_start = 0, box_end = 1000):
    """
    Mutates arr.
    """
    import peakfinder
    def filtfunc2(arr):
        return peakfinder.consolidate_peaks(arr, thr_low = thr_low, thr_high = thr_high, radius = radius)
    def process_island2(subarr):
        subarr[:] -= np.percentile(subarr, 20)
        subarr[:] = filtfunc2(subarr)
        return subarr
    
    # TODO: this is temporary safeguard until I adapt the notebook code
    if detid == 'si':
        raise ValueError
    if detid_match(detid):
        _, clusters = find_islands(detid, box_start, box_end)
    else:
        return arr
    subregions = clustermap(arr, clusters, process_island2)
    for cluster, subarr in zip(clusters, subregions):
        bounding_view(arr, cluster)[:] = subarr
    arr[arr < window_min] = 0
    return arr


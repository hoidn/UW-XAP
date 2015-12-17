import numpy as np
import matplotlib.pyplot as plt
from dataccess import data_access as data
from dataccess import utils

def get_detector_sum_all_events(label, detid, plot = True, nbins = 100):
    """
    Return an array of the integrals of the specified detector accross all
    all events in label.
    """
    path = 'intensity_histograms/' + label + '_' + detid + '.png'
    event_data_getter = lambda arr: np.sum(arr)
    imarray, event_data = data.get_label_data(label, detid,
        event_data_getter = event_data_getter)
    result = np.array(event_data).flatten()
    if plot:
        plt.hist(result, bins = nbins)
        plt.xlabel('Integrated intensity')
        plt.ylabel('number of events')
        plt.savefig(path)
        plt.show()
    return result
        

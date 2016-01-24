import os
import config
import numpy as np
import matplotlib.pyplot as plt
from dataccess import data_access as data
from dataccess import utils


def get_detector_data_all_events(label, detid, funcstr = 'np.sum', plot = True, nbins = 100, filtered = False):
    """
    Evaluate the function event_data_getter (defined in config.py) on all events
    in the dataset and generate a histogram of resulting values.
    """
    def dict_to_list(event_data_dict):
        return utils.merge_dicts(*event_data_dict.values()).values()
        #return reduce(lambda x, y: x + y, event_data_dict.values())
    event_data_getter = eval('config.' + funcstr)
    path = 'intensity_histograms/' + label + '_' + detid + '.png'
    dirname = os.path.dirname(path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(path))
    if not filtered:
        imarray, event_data = data.get_label_data(label, detid,
            event_data_getter = event_data_getter)
    else:
        imarray, event_data = data.get_data_and_filter(label, detid,
            event_data_getter = event_data_getter)
    event_data_list = dict_to_list(event_data)
    result = np.array(event_data_list)#.flatten()
    print "RESULT IS", event_data
    @utils.ifroot
    def plot():
        plt.hist(result, bins = nbins)
        plt.xlabel('output of ' + funcstr)
        plt.ylabel('number of events')
        plt.savefig(path)
        plt.title(label)
        plt.show()
    if plot:
        plot()
    return result

def main(label, detid, funcstr = 'np.sum', nbins = 100, filtered = False):
    
    get_detector_data_all_events(label, detid, funcstr = funcstr, nbins = nbins, filtered = filtered)

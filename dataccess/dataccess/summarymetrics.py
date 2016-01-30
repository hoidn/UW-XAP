import os
import ipdb
import config
import numpy as np
import matplotlib.pyplot as plt
from dataccess import data_access as data
from dataccess import utils


def get_detector_data_all_events(label, detid, funcstr = 'np.sum', func = None, plot = True, nbins = 100, filtered = False, separate = False):
    """
    Evaluate the function event_data_getter (defined in config.py) on all events
    in the dataset and generate a histogram of resulting values.
    """
    @utils.ifroot
    def plot(arr, **kwargs):
        plt.hist(arr, bins = nbins, alpha = 0.5, **kwargs)
        plt.xlabel('output of ' + event_data_getter.__name__)
        plt.ylabel('number of events')
        plt.savefig(path + '.png')
        plt.title('Detector: ' + detid + '; dataset: ' + label)
    @utils.ifroot
    def show():
        plt.show()
    if func is not None:
        event_data_getter = func
    else:
        event_data_getter = eval('config.' + funcstr)
    path = 'histograms/' + label + '_' + detid
    dirname = os.path.dirname(path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(path))
    if not filtered:
        imarray, event_data = data.get_label_data(label, detid,
            event_data_getter = event_data_getter)
    else:
        imarray, event_data = data.get_data_and_filter(label, detid,
            event_data_getter = event_data_getter)
    event_data_list = data.event_data_dict_to_list(event_data)
    if plot:
        if separate:
            for k, d in event_data.iteritems():
                plot(d.values(), label = str(k))
            plt.legend()
        else:
            plot(event_data_list)
        show()
    result = np.array(event_data_list)
    #print "RESULT IS", event_data
    # header kwarg is passed to np.savetxt
    utils.save_0d_event_data(path + '.dat', event_data, header = "Run\tevent\tvalue")
    return result

def main(label, detid, funcstr = 'np.sum', func = None, nbins = 100, filtered = False, **kwargs):
    
    get_detector_data_all_events(label, detid, funcstr = funcstr, func = func, nbins = nbins, filtered = filtered, **kwargs)

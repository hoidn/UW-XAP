import os
import ipdb
import config
import numpy as np
import matplotlib.pyplot as plt
from dataccess import data_access as data
from dataccess import utils

def npsum(arr, **kwargs):
    return np.sum(arr)

def get_detector_data_all_events(labels, detid, funcstr = None, func = None, plot = True, nbins = 100, filtered = False, separate = False):
    """
    Evaluate the function event_data_getter (defined in config.py) on all events
    in the dataset and generate a histogram of resulting values.
    """
    @utils.ifplot
    def plot(arr, label = '', **kwargs):
        try:
            args = data.eventmask_params(label)
        except:
            args = ['()']
        label = (label + "; filter params: %s" % ','.join(map(str, args))) 
#        label = (label + ": mean: %.3f; std: %.3f\n" % (np.mean(arr), np.std(arr)) +
#            "filter params: %s" % ','.join(map(str, args))) 
        plt.hist(arr, bins = nbins, alpha = 0.5, label = label, **kwargs)
    @utils.ifplot
    def finalize_plot():
        plt.xlabel('output of ' + event_data_getter.__name__)
        plt.ylabel('number of events')
        plt.title('Detector: ' + detid)
        plt.legend()
        plt.savefig(merged_path + '.png')
    @utils.ifplot
    def show():
        plt.show()
    if func is not None:
        event_data_getter = func
    else:
        if funcstr is None:
            event_data_getter = npsum
        else:
            event_data_getter = eval('config.' + funcstr)
    basepath = 'histograms/' + detid
    merged_path = (basepath +  '_' + '_'.join(labels))[:100]
    dirname = os.path.dirname(basepath)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(basepath))
    event_data_dicts = []
    if not filtered:
        for label in labels:
            try:
                event_data_dicts.append(data.get_label_data(label, detid, event_data_getter = event_data_getter)[1])
            except ValueError:# no events found in one or more runs in label
                pass
#        event_data_dicts =\
#            [data.get_label_data(label, detid, event_data_getter = event_data_getter)[1]
#            for label in labels]
    else:
        for label in labels:
            try:
                event_data_dicts.append(data.get_data_and_filter(label, detid, event_data_getter = event_data_getter)[1])
            except ValueError:# no events found in one or more runs in label
                print label, ": no events found"
                pass
#        event_data_dicts =\
#            [data.get_data_and_filter(label, detid, event_data_getter = event_data_getter)[1]
#            for label in labels]
    if plot:
        if separate:
            for d, label in zip(event_data_dicts, labels):
                event_data_list = data.event_data_dict_to_list(d)
                plot(event_data_list, label =  label)
        else:
            merged = utils.merge_dicts(*event_data_dicts)
            event_data_list = data.event_data_dict_to_list(merged)
            plot(event_data_list, label = label)
        finalize_plot()
        show()
    result = np.array(event_data_list)
    #print "RESULT IS", event_data
    # header kwarg is passed to np.savetxt
    for d, label in zip(event_data_dicts, labels):
        utils.save_0d_event_data(basepath + '_' + label + '.dat', d, header = "Run\tevent\tvalue")
    utils.save_0d_event_data(merged_path + '.dat', d, header = "Run\tevent\tvalue")
    return result

def main(label, detid, funcstr = None, func = None, nbins = 100, filtered = False, **kwargs):
    
    get_detector_data_all_events(label, detid, funcstr = funcstr, func = func, nbins = nbins, filtered = filtered, **kwargs)

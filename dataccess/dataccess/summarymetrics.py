
import os
import config
import numpy as np

import utils
import query

import playback
from output import rprint

# TODO: refactor this
if config.plotting_mode == 'notebook':
    from dataccess.mpl_plotly import plt
else:
    import matplotlib.pyplot as plt

def npsum(arr, **kwargs):
    return np.sum(arr)

# TODO: handle separate == True with derived dataset labels
def get_detector_data_all_events(labels, detid, funcstr = None, func = None, plot = True, nbins = 100, filtered = False, separate = False):
    """
    Evaluate the function event_data_getter (defined in config.py) on all events
    in the dataset and generate a histogram of resulting values.
    """
    @playback.db_insert
    @utils.ifplot
    def plot(arr, label = '', **kwargs):
        try:
            args = data.eventmask_params(label)
        except:
            args = ['()']
        label = (label + "; filter params: %s" % ','.join(map(str, args))) 
        arr = filter(lambda x: not np.isnan(x), arr)
        plt.hist(arr, bins = nbins, alpha = 0.5, label = label, **kwargs)
    @playback.db_insert
    @utils.ifplot
    def finalize_plot():
        plt.xlabel('output of ' + event_data_getter.__name__)
        plt.ylabel('number of events')
        plt.title('Detector: ' + detid)
        plt.legend()
        plt.savefig(merged_path + '.png')
    @playback.db_insert
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
    def depends_on_data_access():
        from dataccess import data_access as data
        if not filtered:
            for label in labels:
                try:
                    event_data_dicts.append(data.get_label_data(label, detid, event_data_getter = event_data_getter)[1])
                except ValueError:# no events found in one or more runs in label
                    pass
        else:
            for label in labels:
                try:
                    event_data_dicts.append(data.get_data_and_filter(label, detid, event_data_getter = event_data_getter)[1])
                except ValueError:# no events found in one or more runs in label
                    rprint( label, ": no events found")
                    pass
        merged = utils.merge_dicts(*event_data_dicts)
        if plot:
            if separate:
                for d, label in zip(event_data_dicts, labels):
                    event_data_list = data.event_data_dict_to_list(d)
                    plot(event_data_list, label =  label)
            else:
                event_data_list = data.event_data_dict_to_list(merged)
                plot(event_data_list, label = label)
            finalize_plot()
            show()
        result = np.array(data.event_data_dict_to_list(merged))
        #print "RESULT IS", event_data
        # header kwarg is passed to np.savetxt
        for d, label in zip(event_data_dicts, labels):
            utils.save_0d_event_data(basepath + '_' + label + '.dat', d, header = "Run\tevent\tvalue")
        utils.save_0d_event_data(merged_path + '.dat', d, header = "Run\tevent\tvalue")
        return result
    return depends_on_data_access()

def histogram(datasets, detid,  separate = False, event_data_getter = None):
    """
    dataset : query.DataSet
    detid : str
    event_data_getter : function

    Plot a histogram of the output values of event_data_getter evaluated
    over all events in dataset.
    """

    import operator
    def get_eventdata(dataset):
        return dataset.evaluate(detid, event_data_getter = event_data_getter)

    def get_flat_event_data(data_result):
        return data_result.flat_event_data()

    @utils.ifplot
    def plot_hist(arr1d):
        plt.hist(arr1d)

    data_results = map(get_eventdata, datasets)
    if separate:
        [plot_hist(get_flat_event_data(result)) for result in data_results]
    else:
        merged = reduce(operator.add, data_results)
        plot_hist(get_flat_event_data(merged))

def main(dataset_labels, detid, separate = False, funcstr = None, **kwargs):
    import query
    def parse_function_string(function_string, default = None):
        if function_string is None:
            return default
        else:
            return eval('config.' + function_string)
    event_data_getter = parse_function_string(funcstr, utils.usum)
    
    def get_dataset(label):
        return query.existing_dataset_by_label(label)
    datasets = map(get_dataset, dataset_labels)
    @playback.db_insert
    def do_plot():
        histogram(datasets, detid, event_data_getter = event_data_getter, separate = separate)
        plt.show()
    do_plot()

#def main(label, detid, funcstr = None, func = None, nbins = 100, filtered = False, **kwargs):
#    get_detector_data_all_events(label, detid, funcstr = funcstr, func = func, nbins = nbins, filtered = filtered, **kwargs)

import os
import config
import numpy as np
import pdb

import utils
import query

import playback
from output import log

# TODO: refactor this
if config.plotting_mode == 'notebook':
    from dataccess.mpl_plotly import plt
else:
    import matplotlib.pyplot as plt

def npsum(arr, **kwargs):
    return np.sum(arr)

@utils.ifplot
def plot_hist(arr1d, xlabel = '', ylabel = '', label = '', title = '', show = True):
    plt.hist(arr1d, label = label)
    if xlabel:
        plt.xlabel(ylabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if show:
        plt.show()

@utils.ifplot
def plot_scatter(x, y, xlabel = '', ylabel = '', title = '', show = True, **kwargs):
    """
    kwargs are passed through to plt.scatter.
    """
    plt.scatter(x, y, **kwargs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if show:
        plt.show()

def histogram(datasets, detid,  separate = False, event_data_getter = None, show = True):
    """
    Plot a histogram of the output values of event_data_getter evaluated
    over all events in dataset.

    dataset : query.DataSet
    detid : str
    event_data_getter : function
    """

    import operator
    def get_eventdata(dataset):
        return dataset.evaluate(detid, event_data_getter = event_data_getter)

    def get_flat_event_data(data_result):
        return data_result.flat_event_data()


    data_results = map(get_eventdata, datasets)
    if separate:
        labels = [ds.label for ds in datasets]
        [plot_hist(get_flat_event_data(result), label = label, show = False)
            for result, label in zip(data_results, labels)]
        if show:
            plt.show()
    else:
        merged = reduce(operator.add, data_results)
        plot_hist(get_flat_event_data(merged))


def detrend(x, y):
    """
    Do a linear regression, subtract the fit, and return the result.
    """
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return x, y - (slope * x + intercept)

def get_normalized(arr1d):
    """
    Subtract the mean, divide by stdev, and return the result.
    """
    return (arr1d - np.mean(arr1d)) / np.std(arr1d)

def scatter(dataset_identifier, detid_function_1, detid_function_2, normalize = False,
        show = True, frame_processor = None, **kwargs):
    """
    Generate a scatter plot of values returned by eventgetter1 and eventgetter2
    for all events in dataset. Returns two 1d np.ndarrays.

    dataset_identifier : query.DataSet instance or a string equal to the label of an existing
        dataset.
    detid_function_1, detid_function_2 : tuples
        Of format (detector id, function) -> (str, function), denoting a pair of detector
        ID and function with which to evaluate event data.
    
    kwargs are passed through to plt.scatter (if show == True)
    """
    def get_DataSet_instance(ds):
        if isinstance(ds, query.DataSet):
            return ds
        else:
            return query.existing_dataset_by_label(ds)

    detid1, eventgetter1 = detid_function_1
    detid2, eventgetter2 = detid_function_2
    dataset = get_DataSet_instance(dataset_identifier)

    def get_datarun(event_data_getter, detid):
        return dataset.evaluate(detid, event_data_getter = event_data_getter,
                frame_processor = frame_processor)

    def make_label(event_data_getter, detid):
        return "Function: %s; detector: %s" % (event_data_getter.__name__, detid)

    def delete_mismatching(dr1, dr2):
        """
        Return DataRun instances containing only the intersection of events
        in dr1 and dr2.
        """
        if dr1.nevents() != dr2.nevents():
            log("Warning: missing events will be dropped.")
        return dr1.matching_flat_event_data(dr2), dr2.matching_flat_event_data(dr1)

    def get_event_values(datarun):
        raw = datarun.flat_event_data()
        if normalize:
            return get_normalized(raw)
        return raw

    data1, data2 = delete_mismatching(
            *map(get_datarun, [eventgetter1, eventgetter2], [detid1, detid2]))
    xlabel, ylabel = map(make_label, [eventgetter1, eventgetter2], [detid1, detid2])
    plot_scatter(data1, data2, xlabel = xlabel, ylabel = ylabel,
            show = show, **kwargs)
    return data1, data2
    

def main(dataset_labels, detid, separate = False, func = None, **kwargs):
    """
    TODO
    """
    def parse_function_string(function, default = None):
        """
        Function may be either a function or a string that evaluates to one.
        """
        if function is None:
            return default
        elif isinstance(function, str):
            return eval('config.' + function)
        else:
            return function
    event_data_getter = parse_function_string(func, utils.usum)
    
    def get_dataset(label):
        return query.existing_dataset_by_label(label)
    datasets = map(get_dataset, dataset_labels)
    @playback.db_insert
    def do_plot():
        histogram(datasets, detid, event_data_getter = event_data_getter, separate = separate)
        plt.show()
    do_plot()


from dataccess import xrd
from dataccess import query
from dataccess import utils


"""
Utility functions for beam run lp70. plot_run_events is the primary public interface.
"""

def make_event_mask(runs, events_included, max_events = 1000):
    """
    Given a list of run numbers, and a list of list of event numbers to be included for each
    run, return an event mask.
    """
    mask = {}
    for r, events in zip(runs, events_included):
        mask[r] = {}
        for i in range(max_events):
            if i in events:
                mask[r][i] = True
            else:
                mask[r][i] = False
    return mask

def make_pattern_getter(detid, peakfinder = False, compound_list = None):
    def pattern_getter(arr, **kwargs):
        return xrd.Pattern.from_dataset(arr, detid, compound_list)
    def pattern_getter_peakfitting(arr, **kwargs):
        from dataccess.peakfinder import peakfilter_frame
        return xrd.Pattern.from_dataset(peakfilter_frame(arr, detid = detid), detid, compound_list)
    if peakfinder:
        return pattern_getter_peakfitting
    return pattern_getter

def plot_patterns(patterns, labels= None, show = True):
    if labels is None:
        labels =  [''] * len(patterns)
    ax = None
    for pat, label in zip(patterns[:-1], labels[:-1]):
        ax, _ = pat.plot(ax = ax, show = False, label = label)
    ax, _ = patterns[-1].plot(ax = ax, show = show, label = labels[-1])

def plot_run_events(detid, runs, events, plot_mean = False, plot_individual  = True,
        show = True, peakfinder = False, compound_list = None):
    """
    Plot powder patterns of a dataset, consisting of a set of events belonging to one or
    more runs.

    This function returns a list of instances of xrd.Pattern.

    Arguments: 
    detid : str
        Detector identifier (a key string in config.detinfo_map)
    runs : list of ints
        List of one or more run numbers
    events : list of lists of ints
        Events to process for each run. If None, all events are processed
        (compatible only with plot_individual == False). If not None, the
        length of this list must be equal to that of runs.
    plot_mean : bool
        Calculate the mean powder pattern over specified events.
    plot_individual : bool
        Calculate the powder pattern for each individual specified event.
    peakfinder : bool
        If True, do photon hit finding (only works in the low-intensity regime)
    compound_list : list of strings
        List of compounds contributing to the XRD signal (must be specified in config.powder_angles). For the purpose of visualization and estimation of peak parameters using methods of the classes xrd.Peak and xrd.PeakParams.
    """
    ds = query.DataSet(runs, label = str(runs) + str(events))
    pattern_getter = make_pattern_getter(detid, peakfinder = peakfinder, compound_list = compound_list)
    if plot_individual:
        evaluated = ds.evaluate(detid, event_data_getter = pattern_getter,
                                    event_mask = make_event_mask(runs, events))
    else:
        if events:
            evaluated = ds.evaluate(detid, event_mask = make_event_mask(runs, events))
        else:
            evaluated = ds.evaluate(detid)
    ax = None
    output = []
    if plot_mean:
        pat = xrd.Pattern.from_dataset(evaluated.mean, detid, compound_list)
        output.append(pat)
        if not plot_individual:
            ax, _ = pat.plot(ax = ax, show = show, label = 'runs %s: mean' % str(runs))
        else:
            ax, _ = pat.plot(ax = ax, show = False, label = 'runs %s: mean' % str(runs))

    if plot_individual:
        for r, events in zip(runs, events):
            for event in events:
                pat = evaluated.event_data[r][event]
                output.append(pat)
                if event == events[-1] and r == runs[-1]:
                    ax, _ = pat.plot(ax = ax, show = show, label = 'run %s, event %s' % (r, event))
                else:
                    ax, _ = pat.plot(ax = ax, show = False, label = 'run %s, event %s' % (r, event))
    return output

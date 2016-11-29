# Redirect stdout to file mecana.log
from IPython import get_ipython
ipython = get_ipython()

import config
config.stdout_to_file = True

from dataccess import xrd
from dataccess import query
from dataccess import utils
import matplotlib.pyplot as plt
import numpy as np
import os

from dataccess import logbook
from itertools import *
from collections import namedtuple
from dataccess import utils
import pdb

# Configure notebook graphics
#%matplotlib inline
#import mpld3
#from mpld3 import plugins
#mpld3.enable_notebook() 

#from plotly.offline import init_notebook_mode, iplot_mpl
#init_notebook_mode()

# TODO: housekeeping

# Disable deprecation warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def get_query_dataset(querystring, alias = None, print_enable = True, **kwargs):
    dataset = query.main(querystring.split(), label = alias, **kwargs)
    if utils.isroot() and print_enable:
        print "Runs: ", dataset.runs
    return dataset

def get_query_label(*args, **kwargs):
    return get_query_dataset(*args, **kwargs).label

def run_xrd(query_labels, optionstring = '', bsub = False, queue = 'psfehq'):
    prefix_sc = 'mecana.py xrd quad2 -l '
    labelstring = ' '.join(query_labels)
    print prefix_sc, labelstring, optionstring
    if bsub:
        os.system('bsub -n 8 -a mympi -q %s -o batch_logs/%%J.log mecana.py -n xrd quad2 -l %s %s' % \
                  (queue, labelstring, optionstring))
    else:
        #%px %run mecana.py -n xrd quad2 -l $labelstring $optionstring
        from dataccess import xrd
        ipython.magic("run %s %s %s" % (prefix_sc, labelstring, optionstring))
# TODO: caching prevents the plotting of peak fits on repeated calls

def xrd_param(xrdset):
    for p in xrdset.patterns:
        peaks = p.peaks.peaks
        yield [pk.param_values for pk in peaks]

def mean_params(xrdset):
    params = list(xrd_param(xrdset))
    keys = params[0][0].keys()
    def get_mean(key):
        return np.mean([[p[key] for p in pattern] for pattern in params], axis = 0)
    return {k: get_mean(k) for k in keys} 


def print_xrd_param(xrdset, param):
    for p in xrdset.patterns:
        peaks = p.peaks.peaks
        print [pk.param_values[param] for pk in peaks]

def eval_xrd(datasets, compound_list, x = None, detectors = ['quad2'], normalization = 'peak', bgsub = False, iter_parameters = {}, globally_fixed_params = [], **kwargs):
    """
    set_parameters: {param name: [list of peak indices]}

    If uniform_parameters is not empty, two iterations of peak fitting are
    performed. The keys are parameters strings mapping to a list of peak
    indices for which the parameter will be constrained to the mean value among
    all peaks from the first iteration.
    """
    x = xrd.XRD(detectors, datasets,  compound_list = compound_list, bgsub = bgsub, **kwargs)
            #fixed_params = globally_fixed_params, **kwargs)
    for param, indices in iter_parameters.iteritems():
        values = mean_params(x)[param]
        for i in indices:
            x.set_model_param(param, values[i], i)
    return x


def plot_xrd(datasets, compound_list, iter_parameters = {}, plot_progression = False,
        plot_patterns = False, normalization = 'peak', show = True, **kwargs):
    x = eval_xrd(datasets, compound_list, iter_parameters = iter_parameters, **kwargs)
    def get_label(ds):
        return ds.label
    if plot_progression and not plot_patterns:
        x.plot_progression(show = show, normalization = normalization, **kwargs)
    elif plot_patterns and not plot_progression:
        x.plot_patterns(normalization = normalization, show = show, **kwargs)
    else: # plot both
        x.plot_patterns(normalization = normalization, show = show, **kwargs)
        #return x.plot_progression(show = show, **kwargs)
        x.plot_progression(normalization = normalization, show = show, **kwargs)
    if show:
        xrd.plt.show()
    return x

def make_summary_table():
    from IPython.display import HTML, display

    def display_table(title, data):
         display(HTML(
            '<h2>%s</h2>' % title +
            '<table><tr>{}</tr></table>'.format(
                '</tr><tr>'.join(
                    '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
                )
         ))

    table_materials = ['Fe3O4', 'Fe3O4HEF', 'MgO', 'MgOHEF', 'Graphite']
    table_materials_labels = ['Fe$_{3}$O$_4$ ref', 'Fe$_3$O$_4$ HEF', 'MgO ref', 'MgO HEF', 'Graphite ref']
    table_headings = ['1C short pulse', '1C long pulse', '2C short delay (< 70 fs)', '2C long delay (> 70 fs)']

    def get_row_datasets(material):
        """
        Return list of datasets corresponding to each cell in one row of the table
        """
        def merge_queries(query_string_list):
            dsets = [get_query_dataset(q, print_enable=False) for q in query_string_list]
            if len(dsets) == 1:
                return dsets[0]
            elif len(dsets) == 2:
                label_combined = ''.join(map(lambda x: x.label, dsets))
                return dsets[0].union(dsets[1], label_combined)

        querystring_lists =\
            [['material %s runs 870 961' % material, 'material %s runs 1071 1136' % material],
             ['material %s runs 1016 1070' % material],
            ['material %s runs 0 870 delay 0 80' % material],
            ['material %s runs 0 870 delay 80 200' % material]]
        datasets = map(merge_queries, querystring_lists)
        #print 'output:'
        result = [ds.runs for ds in datasets]
        return result
        #for ds in datasets:
        #    print ds.runs

    def get_row_numbers(material):
        return np.array(map(len, get_row_datasets(material)))

    col1 = np.array([''] + table_materials_labels)[:, np.newaxis]
    rest = np.vstack(((np.array(table_headings)), np.array(map(get_row_numbers, table_materials))))
    table = np.hstack((col1, rest))
    display_table('LK20: breakdown of runs by type', table)
    

def accumulate(operator, initial, sequence):
    if not sequence:
        return initial
    return operator(sequence[0], accumulate(operator, initial, sequence[1:]))
def accumulate_sum(sequence):
    def operator(elt, lst):
        if not lst:
            return [elt]
        return [elt + lst[0]] + lst
    return accumulate(operator, [], sequence[::-1])[::-1]
accumulate_sum(range(10))

@utils.memoize(None)
def filter_flux(label):
    try:
        return bool(logbook.get_label_attribute(label, 'focal_size') and logbook.get_label_attribute(label, 'transmission'))
    except:
        return False
    
Run = namedtuple('Run', ['run', 'group', 'focal_size', 'transmission'])
def partition_runs(dataset):
    """
    Return a list of tuples containing start and end values of ranges
    of run numbers that partition the dataset into consecutive sequences
    of runs with non-descending focal spot sizes.
    """
    runs = map(int, filter(filter_flux, map(str, dataset.runs)))
    focal_sizes = [logbook.get_label_attribute(r, 'focal_size') for r in filter(filter_flux, runs)]
    transmissions = [logbook.get_label_attribute(r, 'transmission') for r in filter(filter_flux, runs)]
    
    def new_group(index):
        if index >= len(runs) - 1 or index == 0:
            return 0
        else:
            return int(runs[index] - runs[index - 1] > 10)
    groups = accumulate_sum(map(new_group, range(len(focal_sizes))))
    
    run_tuples = [Run(run, group, focal_size, transmission) for run, group in zip(runs, groups, focal_sizes, transmissions)]
    
    groups = groupby(run_tuples, lambda rt: rt.group)
    return [[rt for rt in rts] for _, rts in groups]

def get_progressions(dataset):
    """
    Return list of groups of run numbers of at least two runs.
    
    A group is defined as a collection of run numbers with a maximum difference
    of 10 between a given run number and the element of the remaining values closest
    to it.
    """
    return filter(lambda runlist: len(runlist) > 1, partition_runs(dataset))

def batch_preprocess(run_number):
    os.system(r'bsub -n 1 -q psfehq -o batch_logs/%J.log ' + 'mecana.py -n datashow quad2 ' + str(run_number))

def bashrun(cmd):
    import subprocess
    return subprocess.check_output(cmd, shell=True)

def batch_submit(cmd, ncores = 6, queue = 'psfehq'):
    return bashrun(r'bsub -n {0} -q {1} -o batch_logs/%J.log mpirun '.format(ncores, queue) + cmd)

def ipcluster_submit(name, ncores = 6, queue = 'psfehq'):
    """Launch an ipcluster in MPI mode as a batch job"""
    return batch_submit(r"ipcluster start --n={0} --profile={1} --ip='*' --engines=MPI".format(ncores, name), ncores = ncores, queue = queue)

def bjobs_count(queue_name):
    return int(bashrun('bjobs -q %s -u all -p | wc -l' % queue_name))

class Bqueue:
    sizes = {'psanaq': 960., 'psfehq': 288., 'psnehq': 288.}
    def __init__(self, name):
        self.name = name
        if name == 'psfehq' or name == 'psnehq':
            self.upstream_q = Bqueue(name[:-1] + 'hiprioq')
        else:
            self.upstream_q = None
        
    def number_pending(self):
        if self.upstream_q is not None:
            return bjobs_count(self.name) + bjobs_count(self.upstream_q.name)
        else:
            return bjobs_count(self.name)
        
    def key(self):
        """
        Sorting key for this class.
        """
        return self.number_pending()/Bqueue.sizes[self.name]

usable_queues = map(Bqueue, config.queues)

def best_queue():
    """Return the name of the least-subscribed batch queue"""
    return min(usable_queues, key = lambda q: q.key())


def preprocess_run(run_number, cores = 1, detid = 'quad2'):
    cmd = r'bsub -n {0} -o batch_logs/%J.log -q {1} mpirun mecana.py -n datashow {2} {3}'.format(cores, best_queue().name, detid, run_number)

    #cmd = r'bsub -a mympi -n {0} -q {1} -o batch_logs/%J.log mecana.py -n datashow {2} {3}'.format(cores, best_queue(), detid, run_number)
    print 'submitting: %s' % cmd
    os.system(cmd)

def preprocess_dataset(dataset, detid, cores = 1):
    [preprocess_run(run, cores = cores, detid = detid)
        for run
        in dataset.runs]
    
def preprocess_xrd(ds, cores = 1):
    [preprocess_run(run) for run in ds.runs]
    os.system(r'bsub -a mympi -n {0} -q {1} -o batch_logs/%J.log mecana.py -n xrd quad2 -l {2} -n peak -b'.format(
              cores, best_queue(), ds.label))

def preprocess_histogram(ds, detid, event_data_getter_name = '', ncores = 1):
    """
    ds: DataSet or string
    """
    if isinstance(ds, str):
        ds = query.existing_dataset_by_label(ds)
    import os
    if event_data_getter_name:
        edg_opt = ' -u ' + event_data_getter_name
    else:
        edg_opt = ''
    cmd = r'mecana.py -n histogram {0} {1}'.format(detid, ds.label) + edg_opt
    batch_submit(cmd, ncores = ncores, queue = best_queue().name)

def get_run_strings(dslist):
    import operator
    return map(str, reduce(operator.add, [ds.runs for ds in dslist]))

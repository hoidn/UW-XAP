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
        import os
        os.system('bsub -n 8 -a mympi -q %s -o batch_logs/%%J.log mecana.py -n xrd quad2 -l %s %s' % \
                  (queue, labelstring, optionstring))
    else:
        #%px %run mecana.py -n xrd quad2 -l $labelstring $optionstring
        from dataccess import xrd
        ipython.magic("run %s %s %s" % (prefix_sc, labelstring, optionstring))
        
def plot_xrd(datasets, compound_list, normalization = 'peak', plot_progression = False, bgsub = False, plot_patterns = False):
    def get_label(ds):
        return ds.label
    if plot_progression and not plot_patterns:
        x = xrd.XRD(['quad2'], map(get_label, datasets),  compound_list = compound_list, bgsub = bgsub, normalization = normalization,\
            plot_progression = True, plot_peakfits = False)
        x.plot_progression(show = True)
    elif plot_patterns and not plot_progression:
        x = xrd.XRD(['quad2'], map(get_label, datasets),  compound_list = compound_list, bgsub = bgsub, normalization = normalization,\
            plot_progression = False, plot_peakfits = True)
        x.plot_patterns()
        xrd.plt.show()
    else: # plot both
        x = xrd.XRD(['quad2'], map(get_label, datasets),  compound_list = compound_list, bgsub = bgsub, normalization = normalization,\
            plot_progression = False, plot_peakfits = True)
        x.plot_patterns()
        x.plot_progression(show = True)
        
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
    import os
    os.system(r'bsub -n 1 -q psfehq -o batch_logs/%J.log ' + 'mecana.py -n datashow quad2 ' + str(run_number))

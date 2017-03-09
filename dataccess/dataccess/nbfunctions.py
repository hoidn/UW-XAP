# Redirect stdout to file mecana.log
from IPython import get_ipython
ipython = get_ipython()

import config
config.stdout_to_file = True

from dataccess import xrd
from dataccess import query
from dataccess import utils
import numpy as np
import os

from dataccess import api
from dataccess import logbook
from itertools import *
from collections import namedtuple
import pdb
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter as gfilt


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

def eval_xrd(datasets, compound_list, x = None, detectors = ['quad2'], normalization = 'peak', bgsub = False, iter_parameters = {}, param_initial_values = {}, globally_fixed_params = [], peak_widths = {}, **kwargs):
    """
    set_parameters: {param name: [list of peak indices]}

    If uniform_parameters is not empty, two iterations of peak fitting are
    performed. The keys are parameters strings mapping to a list of peak
    indices for which the parameter will be constrained to the mean value among
    all peaks from the first iteration.
    """
    x = xrd.XRD(detectors, datasets,  compound_list = compound_list, bgsub = bgsub, fit = False,
            **kwargs)
            #fixed_params = globally_fixed_params, **kwargs)
    for index in param_initial_values:
        for param in param_initial_values[index]:
            value, fixed = param_initial_values[index][param]
            x.set_model_param(param, value, index, fixed = fixed)
    x.fit_all()
    if peak_widths:
        for i in peak_widths:
            def set_width(peak):
                peak.peak_width = peak_widths[i]
                return peak
            x.map_peak(i, set_width)

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
    if event_data_getter_name:
        edg_opt = ' -u ' + event_data_getter_name
    else:
        edg_opt = ''
    cmd = r'mecana.py -n histogram {0} {1}'.format(detid, ds.label) + edg_opt
    batch_submit(cmd, ncores = ncores, queue = best_queue().name)

def get_run_strings(dslist):
    import operator
    return map(str, reduce(operator.add, [ds.runs for ds in dslist]))

def make_powder_pattern_cm(bgsub = False, bg_pattern = None):
    def powder_pattern_cm(imarr = None, bgsub = False, **kwargs):
        import numpy as np
        from dataccess.peakfinder import peakfilter_frame
        from dataccess import xrd
        from dataccess.output import log
        #pdb.set_trace()
        dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
        if bgsub:
            assert bg_pattern is not None
            log('mean value: ', np.mean(dss.intensities))
            log('mean frame value: ', np.mean(dss.image))
            log('mean value of bg pattern:', np.mean(bg_pattern.intensities))
            return np.array(dss.centers_of_mass(bg_pattern = bg_pattern))
        else:
            return np.array(dss.centers_of_mass())
    return powder_pattern_cm

powder_pattern_cm = make_powder_pattern_cm(bgsub = False)

#def peak_cm(ds, bgsub = True):
#    def powder_pattern_single_frame(imarr = None, **kwargs):
#        import numpy as np
#        from dataccess.peakfinder import peakfilter_frame
#        from dataccess import xrd
#        dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
#        return np.array([dss.angles, dss.intensities])
#    #TODO: WHAT's up with the normalization here?
#    #my_xrd = xrd.XRD(['quad2'], [ds], compound_list=['MgO'], bgsub = False, fit = False,
#    #                 frame_processor = peakfilter_frame)
#    bg_result = ds.evaluate('quad2', frame_processor = powder_pattern_single_frame,
#                            event_data_getter = utils.identity)
#    bg_angles, bg_intensity = bg_result.mean
#    bg_pattern = xrd.Pattern(bg_angles, bg_intensity, ['MgO'])
#
#    return powder_pattern_cm, bg_pattern

def get_cm_frame_processor(bgsub = True, bg_pattern = None):
    def powder_pattern_cm(imarr = None, nevent = None,
            **kwargs):
        """
        Return peak centers of mass for a single event's powder pattern.
        """
        from dataccess.peakfinder import peakfilter_frame
        from dataccess.output import log
        dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
        if bgsub:
            assert bg_pattern is not None
            log('mean value: ', np.mean(dss.intensities))
            log('mean frame value: ', np.mean(dss.image))
            log('mean value of bg pattern:', np.mean(bg_pattern.intensities))
            uncorrected_cm = np.array(dss.centers_of_mass(bg_pattern = bg_pattern))
        else:
            uncorrected_cm = np.array(dss.centers_of_mass())
        return uncorrected_cm 
    return powder_pattern_cm

def cm_frame_processor(imarr = None, nevent = None,
        **kwargs):
    """
    Return peak centers of mass for a single event's powder pattern.
    """
    from dataccess.peakfinder import peakfilter_frame
    from dataccess.output import log
    dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
    return np.array(dss.centers_of_mass())

def powder_pattern_single_frame(imarr = None, **kwargs):
    from dataccess.peakfinder import peakfilter_frame
    dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
    return np.array([dss.angles, dss.intensities])

class CM_Interpolation:
    def __init__(self, func, sourcex, sourcey):
        self.func = func
        self.x = sourcex
        self.y = sourcey
    def __call__(self, x):
        return self.func(x)

def interpolated_cms(ds, bgsub = False, sigma_max = 3., subtract_mean = True, peakfit_recenter = True):
    """
    Return smoothed interpolation of the mean-subtracted center of mass
    positions for for all MgO powder peaks.  Note that no correction is applied
    for the background's influence on the calculated center of mass.
    """
    import numpy.ma as ma
    def make_interp(i, smoothing = 20):
        peak_angles = get_peak_angles(ds)
        if sigma_max > 0:
            #pdb.set_trace()
            if peakfit_recenter:
                mask = np.abs(cms - peak_angles) < sigma_max * np.std(cms, axis = 0)
            else:
                mask = np.abs(cms - np.mean(cms, axis = 0)) < sigma_max * np.std(cms, axis = 0)
            goodcms = ma.masked_array(cms, ~mask)
            data, mask = goodcms[:, i].data, goodcms[:, i].mask
            good_events, good_cms = event_numbers[~mask], data[~mask]
        else:
            good_events, good_cms = event_numbers, good_cms
        if subtract_mean:
            if peakfit_recenter:
                good_cms = good_cms - peak_angles[i]
            else:
                good_cms = good_cms - np.mean(good_cms)
        return CM_Interpolation(utils.extrap1d(interp1d(good_events, gfilt(good_cms, smoothing))),
                                good_events, good_cms)

    bg_result = ds.evaluate('quad2', frame_processor = powder_pattern_single_frame,
                            event_data_getter = utils.identity)
    bg_angles, bg_intensity = bg_result.mean
    bg_pattern = xrd.Pattern(bg_angles, bg_intensity, ['MgO'])
    def to_pattern(angles, intensities):
        return xrd.Pattern(angles, intensities, compound_list = ['MgO'], label = 'test')

    trace = ds.evaluate('quad2', frame_processor = get_cm_frame_processor(bg_pattern = bg_pattern, bgsub = bgsub),
                        event_data_getter = utils.identity)
    event_numbers, cms = np.arange(len(trace.flat_event_data())), trace.flat_event_data()
#    pattern_result = ds.evaluate('quad2', frame_processor = powder_pattern_single_frame,
#                        event_data_getter = utils.identity)
#    from dataccess.psget import get_pool
#    pool = get_pool()
#    cms = np.array(pool.map(
#        lambda angles_intensities: np.array(to_pattern(*angles_intensities).centers_of_mass()),
#        list(pattern_result.iter_event_value_pairs())))
    event_numbers = np.array(range(len(cms)))

    return [make_interp(i) for i in range(cms.shape[1])]

#def interpolated_cms(ds, bgsub = False, sigma_max = 3.):
#    """
#    Return smoothed interpolation of the mean-subtracted center of mass
#    positions for for all MgO powder peaks.
#    """
#    import numpy.ma as ma
#    def make_interp(i, smoothing = 20):
#        if sigma_max > 0:
#            mask = np.abs(cms - np.mean(cms, axis = 0)) < sigma_max * np.std(cms, axis = 0)
#            goodcms = ma.masked_array(cms, ~mask)
#            data, mask = goodcms[:, i].data, goodcms[:, i].mask
#            good_events, good_cms = event_numbers[~mask], data[~mask]
#        else:
#            good_events, good_cms = event_numbers, good_cms  
#        #return event_numbers[~mask], data[~mask]
#        return utils.extrap1d(interp1d(good_events, gfilt(good_cms - np.mean(good_cms), smoothing)))
#
#    bg_result = ds.evaluate('quad2', frame_processor = powder_pattern_single_frame,
#                            event_data_getter = utils.identity)
#    bg_angles, bg_intensity = bg_result.mean
#    bg_pattern = xrd.Pattern(bg_angles, bg_intensity, ['MgO'])
#    trace = ds.evaluate('quad2', frame_processor = get_cm_frame_processor(bg_pattern = bg_pattern, bgsub = bgsub),
#                        event_data_getter = utils.identity)
#    event_numbers, cms = np.arange(len(trace.flat_event_data())), trace.flat_event_data()
#    return [make_interp(i) for i in range(cms.shape[1])]
#    if correct_cm:
#        raise NotImplementedError
#        return np.array(
#            [np.array(
#                    [uncorrected - smooth_cm_200_interp_nosub(nevent, i),
#                     uncorrected - smooth_cm_200_interp(nevent, i), uncorrected])
#             for i, uncorrected in enumerate(uncorrected_cm)])
#    else:
#        return uncorrected_cm

# TODO: this isn't quite right, since it uses the default peak locations (not the fitted values)
def get_peak_angles(dataset, mode = 'mean', n_iters = 2):
    """
    Return the positions (in deg) of powder peaks for the given dataset
    """
    uncorrected = dataset.evaluate('quad2', frame_processor = powder_pattern_single_frame,
                               event_data_getter = utils.identity)
    angles, intensities = uncorrected.mean
    pat = xrd.Pattern(angles, intensities, compound_list = ['MgO'])
    if n_iters <= 1:
	peak_angles = np.array(pat.centers_of_mass(mode = mode))
        return peak_angles
    peak_angles = get_peak_angles(dataset, mode = mode, n_iters = n_iters - 1)
    pat = xrd.Pattern(angles, intensities, compound_list = ['MgO'], peak_angles = peak_angles)
    return np.array(pat.centers_of_mass(mode = mode))

def signal_to_background(ds):
    """
    Return signal/background for all peaks in this dataset's powder pattern
    """
    result = ds.evaluate('quad2', frame_processor = powder_pattern_single_frame,
                            event_data_getter = utils.identity)
    angles, intensities = result.mean
    pattern = xrd.Pattern(angles, intensities, ['MgO'], peak_angles = get_peak_angles(ds))
    def one_peak(peak):
        signal = peak.integrate(angles, intensities, mode = 'integral_bgsubbed')
        total = peak.integrate(angles, intensities, mode = 'integral')
        return signal/(total - signal)
    return np.array([one_peak(peak) for peak in pattern.peaks.peaks])

def cm_corrections(ds, fudge_factor = 1.):
    """
    fudge_factor is a scaling parameter for the signal/background ratio.
    """
    interpolators = interpolated_cms(ds)
    ratios = signal_to_background(ds)
    def newfunc(nevent):
        return np.array([(1 + fudge_factor/ratios[i]) * interpolators[i](nevent)[0] for i in range(len(interpolators))])
    return newfunc


#    def newfunc(i):
#        return lambda nevent: (1 + fudge_factor/ratios[i]) * interpolators[i](nevent)


def energy_shifter(pattern, si = None, nevent = None, **kwargs):
    from dataccess import mec
    angles, intensities = pattern.angles, pattern.intensities
    nominal_energy = 8965.
    cm_si = mec.si_imarr_cm_3(si)
    deltaE = cm_si - nominal_energy
    theta_rad = np.deg2rad(angles)
    theta_correction = np.rad2deg(2 * np.tan(theta_rad/2) * deltaE/nominal_energy)
    rescale = (angles + theta_correction)/angles
    angles, intensities = utils.regrid(angles, intensities, rescale)
    return angles, intensities

def wobble_shifter(pattern, si = None, nevent = None, shift_scale = 1., corr = None, **kwargs):
    angles, intensities = pattern.shift_peaks(corr(nevent), shift_scale = shift_scale)
    return angles, intensities

def both_shifter(pattern, si = None, nevent = None, shift_scale = 1., **kwargs):
    angles, intensities = wobble_shifter(pattern, si = si, nevent = nevent, shift_scale = shift_scale, **kwargs)
    return energy_shifter(xrd.Pattern(angles, intensities, compound_list = ['MgO']), si = si, nevent = nevent)

def make_shifter_function(dataset, shifter_func, shift_scale = 1.):
    corr = cm_corrections(dataset)
    @api.register_input_detids('quad2', 'si')
    def shifter(quad2 = None, si = None, **kwargs):
        from dataccess import xrd
        import numpy as np
        from dataccess.peakfinder import peakfilter_frame
        if 'shift_scale' not in kwargs:
            kwargs['shift_scale'] = shift_scale
        cleaned_quad2 = peakfilter_frame(quad2, detid = 'quad2')
        pat = xrd.Pattern.from_dataset(cleaned_quad2, 'quad2', ['MgO'], label  = 'test')
        angles, intensities = shifter_func(pat, si = si, quad2 = quad2, corr = corr, **kwargs)
        return np.array([angles, intensities])
    return shifter

def make_shifter_frame_processor(dataset, pattern_shifter_func):
    shifter = make_shifter_function(dataset, pattern_shifter_func)
    #peak_angles = peak_cms(uncorrected, compound_list = ['MgO'])
    @api.register_input_detids('quad2', 'si')
    def frame_processor(quad2 = None, si = None, **kwargs):
        return shifter(quad2 = quad2, si = si, **kwargs)
    return frame_processor

        #from dataccess.peakfinder import peakfilter_frame
        # TODO set peak angles
        
#        pat = xrd.Pattern.from_dataset(peakfilter_frame(quad2, detid = 'quad2'), 'quad2', ['MgO'],
#                                           label  = 'test')#, peak_angles = peak_angles)

#def peak_cm_shifted(bg_result, bgsub = True, correct_cm = False):
#    from dataccess.output import log
#    import numpy as np
#    from dataccess.peakfinder import peakfilter_frame
#    from dataccess import xrd
#    from dataccess.nbfunctions import eval_xrd
#
#    def smooth_cm_200_interp_nosub(n_evt, i):
#        cms_nosub, smoothed_cms_nosub, smoothed_cms_nosub_interp, trace = cm_variation_scatter(ds, bgsub=False)
#        sampled = smoothed_cms_nosub_interp(n_evt)[:, i]
#        if hasattr(n_evt, '__iter__'):
#            return sampled
#        else:
#            return sampled[0]
#
#    def smooth_cm_200_interp(n_evt, i):
#        cms, smoothed_cms, smoothed_cms_interp, trace = cm_variation_scatter(ds, bgsub=True)
#        sampled = smoothed_cms_interp(n_evt)[:, i]
#        if hasattr(n_evt, '__iter__'):
#            return sampled
#        else:
#            return sampled[0]
#
#    
#    #TODO: WHAT's up with the normalization here?
#    #my_xrd = xrd.XRD(['quad2'], [ds], compound_list=['MgO'], bgsub = False, fit = False,
#    #                 frame_processor = peakfilter_frame)
#    bg_angles, bg_intensity = bg_result.mean
#    bg_pattern = xrd.Pattern(bg_angles, bg_intensity, ['MgO'])
#    return powder_pattern_cm, bg_pattern
#
#def cm_variation_scatter(ds, bgsub = True, correct_cm = False):
#    """
#    Returns cms, smoothed_cms, smoothed_cm_interp
#    
#    where cms is a 2d array, each element of which contains the CM coordinates of all powder peaks,
#    and smoothed_cms is a smoothed version of the same. smoothed_cm_interp is the mean-subtracted interpolation function
#    version of smoothed_cms (i.e., it returns angle offset values)
#    """
#    from dataccess.peakfinder import peakfilter_frame
#    def powder_pattern_single_frame(imarr = None, **kwargs):
#        dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
#        return np.array([dss.angles, dss.intensities])
#    bg_result = ds.evaluate('quad2', frame_processor = powder_pattern_single_frame,
#                            event_data_getter = utils.identity)
#    trace = ds.evaluate('quad2', frame_processor = peak_cm_shifted(bg_result, bgsub = bgsub, correct_cm = correct_cm)[0], event_data_getter = utils.identity)
#    event_numbers, cms = np.arange(len(trace.flat_event_data())), trace.flat_event_data()
#    smoothed_cm_interp = utils.extrap1d(interp1d(event_numbers, gfilt(cms - np.mean(cms, axis = 0), sigma = [20, 0]),
#                                                 axis = 0))
#    smoothed_cms = smoothed_cm_interp(event_numbers)
#    return cms, smoothed_cms, smoothed_cm_interp, trace
#    #cms_highpass = cms - smoothed_cms
#    #from dataccess.mec import si_imarr_cm_3
#    #energies = ds.evaluate('si', frame_processor = si_imarr_cm_3, event_data_getter = utils.identity)
#
#    #plt.scatter(event_numbers, cms, label = 'raw')
#    #plt.plot(np.array(event_numbers), smoothed_cms, label  ='interpolation, smoothed with sigma = 20')
#    #plt.show()

# Authors: A. Ditter, O. Hoidn, and R. Valenza

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import copy
import pdb
import operator

import config
import utils
import playback
from output import log
import geometry

if config.plotting_mode == 'notebook':
    from dataccess.mpl_plotly import plt
else:
    import matplotlib.pyplot as plt


def _all_equal(lst):
    """
    Return true if all elements of a list are equal
    """
    s = set(lst)
    if len(s) <= 1:
        return True
    return False


def _patterns_compound(patterns):
    compound_list = [pattern.get_compound() for pattern in patterns]
    if not _all_equal(compound_list):
        raise ValueError("pattern compounds missing, or mismatching")
    return compound_list.pop()

def peak_progression(patterns, normalization = None,
        peak_width = config.peak_width, fixed_params = [], **kwargs):
    """
    Note: this function may only be called if the elements of labels are
    google spreadsheet dataset labels, rather than paths to data files.
    """
    if normalization is None:
        normalization = 'transmission'

    compound_name = _patterns_compound(patterns)
    # sort by increasing beam intensity

    def get_flux_density(pat):
        #TODO: make this function take a dataref. move config.py functions into another file
        beam_energy = config.beam_intensity_diagnostic(pat.label)
        size = pat.get_attribute('focal_size')
        # convert length units from microns to cm
        return beam_energy / (np.pi * ((size * 0.5 * 1e-4)**2))

    patterns = sorted(patterns, key = get_flux_density)
    # TODO: fix peak progression calculation for two-detid mode
    powder_angles = np.array(geometry.get_powder_angles(compound_name))

    label_flux_densities = map(get_flux_density, patterns)

    # indices: label, peak
    peaksize_array = np.array([pat.peak_sizes() for pat in patterns])
    normalized_peaksize_array = peaksize_array / get_normalization(patterns,
        peak_width = peak_width, type = normalization, fixed_params = fixed_params,
        **kwargs)[:, np.newaxis]

    # indices: peak, label
    heating_progression = normalized_peaksize_array.T
    normalized_heating_progression = heating_progression / heating_progression[:, 0][:, np.newaxis]
    return powder_angles, label_flux_densities, heating_progression, normalized_heating_progression

def plot_peak_progression(patterns, maxpeaks = 'all', ax = None, logscale = True,
        normalization = None, show = False, inner_normalization = True, fixed_params = [],
        **kwargs):

    powder_angles, pattern_fluxes, progression, normalized_progression =\
            peak_progression(patterns, normalization = normalization,
            fixed_params = fixed_params )

    output = []
    # TODO: reimplement selection of most intense peaks
    def plotting(ax):
        if ax is None:
            f, ax = plt.subplots(1)
        if logscale:
            ax.set_xscale('log')
	if inner_normalization:
	    for label, curve in zip(map(str, powder_angles), normalized_progression):
		ax.plot(pattern_fluxes, curve, label = label)
		output.append(curve)
	else:
	    for label, curve in zip(map(str, powder_angles), progression):
		ax.plot(pattern_fluxes, curve, label = label)
		output.append(curve)
        plt.legend()
        ax.set_xlabel('Flux density (J/cm^2)')
        ax.set_ylabel('Relative Bragg peak intensity')
        #ax.set_xlim((min(pattern_fluxes), max(pattern_fluxes)))
        if show:
            plt.show()
    plotting(ax)
    return output

# TODO: why does memoization fail?
#@utils.eager_persist_to_file("cache/xrd.process_dataset/")
def process_dataset(dataset, nbins = 1000, fiducial_ellipses = None,
        bgsub = True, **kwargs):
    pat = Pattern(dataset, nbins = nbins, fiducial_ellipses = fiducial_ellipses,
            bgsub = bgsub, **kwargs)
#    def process_one_detid(detid):
#        imarray = dataset.get_array(detid)#data_extractor(dataset, **kwargs)
#        return geometry.process_imarray(detid, imarray,
#            fiducial_ellipses = fiducial_ellipses, bgsub = bgsub,
#            compound_list = dataset.compound_list, **kwargs)
    return pat.angles, pat.intensities
    # return reduce(operator.add, map(process_one_detid, dataset.detid_list))

def main(detid_list, data_identifiers, mode = 'labels', plot = True, plot_progression = False, maxpeaks = 6, **kwargs):
    """
    Arguments:
        detid: id of a quad CSPAD detector
        data_identifiers: a list containing either (1) dataset labels or (2)
            paths to ASCII-formatted data CSPAD data files.
    Keyword arguments:
        mode: == 'labels' or 'paths' depending on the contents of
            data_identifiers
        plot: if True, plot powder pattern(s)
    """
    if mode == 'array':
        labels = ['unknown_' + str(detid_list[0])]
    else:
        labels = data_identifiers
    if plot:
        if plot_progression:
            f, axes = plt.subplots(2)
            ax1, ax2 = axes
        else:
            f, ax1 = plt.subplots()
        xrd = XRD(detid_list, data_identifiers, ax = ax1, plot_progression = plot_progression, **kwargs)
    else:
        xrd = XRD(detid_list, data_identifiers, plot_progression = plot_progression, **kwargs)

    @utils.ifplot
    def doplot(fixed_params = []):
        if plot_progression:
            xrd.plot_progression(ax = ax2, maxpeaks = maxpeaks,
                    fixed_params = fixed_params)
        xrd.plot_patterns(ax = ax1)
        utils.global_save_and_show('xrd_plot/' + '_'.join(detid_list) + '_'.join(labels) + '.png')
    if plot:
        doplot()
    return xrd

class XRDset:
    """
    Represents data on a single detector.
    """
    def __init__(self, dataref, detid, compound_list, mask = True, label = None, **kwargs):
        """
        kwargs are frame-processing related arguments, including
        event_data_getter and frame_processor. See data_access.py for details.
        """
        self.dataref = dataref
        self.detid = detid
        self.compound_list = compound_list
        self.mask = mask
        self.kwargs = kwargs
        if type(dataref) == np.ndarray:
            if label is None:
                raise ValueError("label must be provided for dataset of type numpy.ndarray")
            self.label = label
        else: # dataref must be a 
            self.label = dataref.label


    def get_array(self, event_data_getter = None):
        from dataccess import data_access as data

        #elif dataset.ref_type == 'label':
        if type(self.dataref) == np.ndarray:
            imarray, event_data =  self.dataref, None

        else:# Assume this is a query.DataSet instance
            imarray, event_data = data.eval_dataset_and_filter(self.dataref, self.detid,
                **self.kwargs)
        imarray = imarray.T

        if self.mask:
            extra_masks = config.detinfo_map[self.detid].extra_masks
            combined_mask = utils.combine_masks(imarray, extra_masks, transpose = True)
            imarray *= combined_mask
        min_val =  np.min(imarray)
        if min_val < 0:
            return np.abs(min_val) + imarray
        else:
            return imarray

class RealMask:
    """
    Represents one or more intervals on the real numbers.
    """
    def __init__(self, (start, end), *others):
        """
        Initialize using one or more tuples denoting intervals
        """
        self.intervals = [ (start, end) ]
        for interval in others:
            self.intervals.append(interval)

    def __add__(self, other):
        new_intervals = self.intervals + other.intervals
        return RealMask(new_intervals)

    def includes(self, val):
        """
        Return True if val is within one of the intervals
        """
        for iv in self.intervals:
            if iv[0] <= val <= iv[1]:
                return True
        return False

class Peak:
    """
    Class representing a single powder peak.
    """
    def __init__(self, angle, param_values, fixed_params, peak_width = config.peak_width):
        default_dict = {'amp': 4, 'cen': angle, 'wid': .2}
        self.angle = angle
        self.param_values = {k: param_values.get(k, default_dict[k]) for k in default_dict.keys()}
        self.fixed_params = {k: (k in fixed_params) for k in default_dict.keys()}
        if set(param_values.keys()) != set(fixed_params):
            raise ValueError("Cannot constrain parameters without providing starting values.")
        self.peak_width = peak_width

    def _peak_size_gaussian_fit(self, x, y):
        from lmfit import Model
        def gaussian(x, amp, cen, wid):
            "1-d gaussian: gaussian(x, amp, cen, wid)"
            return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))
        def line(x, slope, intercept):
            "line"
            return slope * x + intercept
        mod = Model(gaussian) + Model(line)
        pars  = mod.make_params( amp=self.param_values['amp'], cen=self.param_values['cen'],
                wid=self.param_values['wid'], slope=0, intercept=1)
        for name, p in pars.iteritems():
            if p.name in self.fixed_params and self.fixed_params[p.name]:
                p.set(vary = False)
        i = self.crop_indices(x, self.peak_width)
        result = mod.fit(y[i], pars, x=x[i])
        best_values = result.best_values
        # The fit values are sometimes inverted, for some reason
        if best_values['amp'] < 0 and best_values['wid'] < 0:
            best_values['amp'] = -best_values['amp']
            best_values['wid'] = -best_values['wid']
        self._post_fit_update_params(best_values)
        return best_values, x[i], result.best_fit

    def _post_fit_update_params(self, best_values):
        for p in self.param_values:
            self.param_values[p] = best_values[p]

    def peak_fit(self, x, y, peak_width = config.peak_width, **kwargs):
        """
        Given a powder pattern x, y, returns a list of tuples of the form (m,
        b, amplitude, xfit, yfit), where:
            m: linear coeffient of background fit
            b: offset of background fit
            amplitude: amplitude of fit's gaussian component
            xfit : np.ndarray. x-values of the fit range
            yfit : np.ndarray. fit evaluated on the array xfit
        """
        x, y = np.array(x), np.array(y)
        from collections import namedtuple
        FitResult = namedtuple('Fit', ['m', 'b', 'amplitude', 'xfit', 'yfit', 'values'])
        best_values, xfit, yfit = self._peak_size_gaussian_fit(x, y) 
        amplitude = best_values['amp']
        m, b = best_values['slope'], best_values['intercept']
        return FitResult(m, b, amplitude, xfit.copy(), yfit.copy(), best_values)

    def crop_indices(self, x, width):
        """ Return powder pattern indices centered around the given angle """
        peakmin = self.angle - self.peak_width/2.
        peakmax = self.angle + self.peak_width/2.
        return np.where(np.logical_and(x >=peakmin, x <= peakmax))[0]


class PeakParams:
    """
    Store a set powder peaks, each consisting of an angle value and of
    peak-fitting parameters.

    Methods:
        fit_peaks : return fit parameters for all peaks, using a guassian +
        linear background model.
    """
    def __init__(self, angles, starting_values = None, fixed_params = None):
        """
        starting_values : list of dicts of form {str -> float}
            Starting values (or, for constrained parameters, final values) for
            one or more of the fit parameters 'wid', 'cen', and 'amp' for the
            gaussian components of peak fits. The default mapping is: {'amp':
            5, 'cen': 5, 'wid': 2}.
        fixed_params : list of tuples of strings
            Specifies which parameters to fix using the values specified in
            starting_values.
        """
        peaks = []
        if starting_values is None:
            starting_values = [{} for _ in angles]
        if fixed_params is None:
            fixed_params = [[] for _ in angles]
        for angle, d, fixed in zip(angles, starting_values, fixed_params):
            peaks.append(Peak(angle, d, fixed))
#        for angle in angles:
#            peaks.append(Peak(angle, starting_values, fixed_params))

        self.peaks = peaks

    def fit_peaks(self, x, y, peak_width = config.peak_width, **kwargs):
        def in_range(angle_degrees):
            return angle_degrees > np.min(x) and angle_degrees < np.max(x)
        valid_peaks = filter(lambda peak: (peak.angle - peak_width/2. >= np.min(x)) and
                        (peak.angle + peak_width/2. <= np.max(x)),
                    self.peaks)
        return [peak.peak_fit(x, y, peak_width = peak_width, **kwargs) for peak in valid_peaks]

    def __add__(self, other):
        import copy
        new = copy.deepcopy(self)
        new.peaks = new.peaks + copy.deepcopy(other.peaks)
        return new


class Pattern:
    def __init__(self, xrdset, peak_width = config.peak_width,
            nbins = 1000, fiducial_ellipses = None, bgsub = False,
            pre_integration_smoothing = 0, starting_values = None,
            fixed_params = None, **kwargs):
        """
        Initialize an instance using a single dataset.
        
        kwargs are passed to Dataset.get_array()
        """
        self.dataset_list = [xrdset]
        self.width = peak_width
        self.compound_list = xrdset.compound_list
        self.angles, self.intensities, self.image =\
                geometry.process_imarray(xrdset.detid, xrdset.get_array(kwargs),
                        compound_list = xrdset.compound_list, nbins = nbins,
                        fiducial_ellipses = fiducial_ellipses, bgsub = bgsub,
                        pre_integration_smoothing = pre_integration_smoothing)
        self.anglemask = RealMask((np.min(self.angles), np.max(self.angles)))
        self.label = xrdset.label
        self.peak_angles = np.array(geometry.get_powder_angles(self.get_compound(),
            filterfunc = lambda ang: self.anglemask.includes(ang)))
        self.peaks = PeakParams(self.peak_angles, starting_values = starting_values,
                fixed_params = fixed_params)

    def get_pattern(self):
        return self.angle, self.intensities

    @classmethod
    def from_dataset(cls, dataset, detid, compound_list, label = None, **kwargs):
        """
        Instantiate using a query.Dataset instance.

        kwargs are passed to Pattern.__init__().
        """
        xrdset = XRDset(dataset, detid, compound_list, label = label)
        return cls(xrdset, label = label, **kwargs)

    def get_attribute(self, attr):
        """
        Return an attribute value for the dataset(s) associated with this instance.
        """
        values = map(lambda ds: ds.dataref.get_attribute(attr), self.dataset_list)
        assert _all_equal(values)
        return values[0]

    @classmethod
    def from_multiple(cls, dataset_list, **kwargs):
        """
        Construct an instance from multiple Dataset instances.
        """
        assert len(dataset_list) >= 1
        components = [Pattern(ds, **kwargs) for ds in dataset_list]
        return reduce(operator.add, components)

    def __add__(self, other):
        """
        Add the data of a second Pattern instance to the current one.
        """
        self.dataset_list = self.dataset_list + other.dataset_list
        assert self.width == other.width
        assert self.compound_list == other.compound_list
        self.angles = self.angles + other.angles
        self.intensities = self.intensities + other.intensities
        self.anglemask = self.anglemask + other.anglemask
        self.peak_angles = np.concatenate((self.peak_angles, other.peak_angles))
        self.peaks = self.peaks + other.peaks
        self.image = np.hstack((self.image, other.image))
        return self

    def normalize(self, normalization):
        """
        Return a new instance with pattern intensity normalized using the given value.
        """
        new = copy.deepcopy(self)
        new.intensities /= normalization
        return new

    def get_compound(self):
        """
        Return the XRD analysis compound (by convension the first element of
        self.dataref.compound_list).
        """
        return self.compound_list[0]

    def _new_ax(self):
        """
        Initialize a plot axis.
        """
        f, axl = plt.subplots(1)
        ax = axl[0]
        ax.set_xlabel('Scattering angle (deg)')
        ax.set_ylabel('Integrated intensity')
        return ax

    def fit_peaks(self, peak_width = config.peak_width):
        return self.peaks.fit_peaks(self.angles, self.intensities, peak_width = peak_width)

    @utils.ifplot
    def plot_peakfits(self, ax = None, show = False, normalization = 1.,
            peak_width = config.peak_width):
        if ax is None:
            ax = self._new_ax()
        for fit in self.fit_peaks():
            ax.plot(fit.xfit, fit.yfit / normalization, color = 'red')
            ax.plot(fit.xfit, (fit.yfit - (fit.m * fit.xfit + fit.b)) / normalization, color = 'black')
        if show:
            plt.show()
        return ax

#    @playback.db_insert
#    @utils.ifplot
    def plot(self, ax = None, label = None, show = False, normalization = None,
            peak_width = config.peak_width, fixed_params = [], legend = False):
        pdb.set_trace()
        if ax is None:
            ax = self._new_ax()
        if normalization:
            scale = get_normalization([ self ], type = normalization,
                    fixed_params = fixed_params)[0]
        else:
            scale = 1.
        if label is None:
            label = self.label
        dthet = (np.max(self.angles) - np.min(self.angles))/len(self.angles)
        ax.plot(self.angles, gaussian_filter(self.intensities, 0.05/dthet)/scale, label = label)
        if normalization == 'peak':
            self.plot_peakfits(ax = ax, show = False, peak_width = peak_width,
                    normalization = scale)
        if legend:
            plt.legend()
        if show:
            plt.show()
        return ax, scale

    def peak_sizes(self, peak_width = config.peak_width, fixed_params = [],
            **kwargs):
        peak_fits = self.peaks.fit_peaks(self.angles, self.intensities, peak_width = peak_width, fixed_params = fixed_params)
        amplitudes = []
        for fit in peak_fits:
            amplitudes.append(fit.amplitude)
        return np.array(amplitudes)




class XRD:
    """
    Class representing one or more datasets and their associated powder
    patterns.

    This class constitutes the user API for XRD analysis.

    kwargs: 
        bgsub: if == 'yes', perform background subtraction; if == 'no',
            don't; and if == 'both', do both, returning two powder patterns
            per element in data_identifiers
        compound_list: list of compound identifiers corresponding to crystals
            for which simulated diffraction data is available.

        Additional kwargs are used to the constructor for XRDset.
    """
    def __init__(self, detid_list, data_identifiers, ax = None, bgsub = False, mode = 'label',
        peak_progression_compound = None, compound_list = [], mask = True,
        plot_progression = False, plot_peakfits = False,
        event_data_getter = None, frame_processor = None, starting_values = None,
        fixed_params = None, **kwargs):

        self.fixed_params = fixed_params

        if bgsub:
            if not compound_list:
                bgsub = False
                log( "No compounds provided: disabling background subtraction.")

        def pattern_one_dataref(dataref):
            def one_detid(detid):
                dataset = XRDset(dataref, detid, compound_list, mask = mask,
                        event_data_getter = event_data_getter, frame_processor = frame_processor,
                        **kwargs)
                return Pattern(dataset, bgsub = bgsub, starting_values = starting_values,
                        fixed_params = fixed_params)
            return reduce(operator.add, map(one_detid, detid_list))

        self.patterns = map(pattern_one_dataref, data_identifiers)

        if not ax:
            self.ax = None
        else:
            self.ax = ax

        for pat in self.patterns:
            pat.fit_peaks()

        
    def show_images(self):
        imarrays = [pat.image for pat in self.patterns]
        labels = [pat.label for pat in self.patterns]
        for im, lab in zip(imarrays, labels):
            utils.show_image(im, title = lab)

    def normalize(self, mode):
        raise NotImplementedError

    def plot_patterns(self, ax = None, normalization = None, show_image = False,
            plot_peakfits = False, **kwargs):
        if ax is None and self.ax is not None:
            ax = self.ax
        elif not ax:
            _, ax = plt.subplots()
        for pat in self.patterns:
            ax, scale = pat.plot(ax = ax, normalization = normalization,
                    fixed_params = self.fixed_params)
        if pat.peak_angles is not None:
            for ang in pat.peak_angles:
                ax.plot([ang, ang], [np.min(pat.intensities)/scale, np.max(pat.intensities)/scale],
                        color = 'black')
        plt.legend()
        if show_image:
            self.show_images()

    def plot_progression(self, ax = None, maxpeaks = 6, normalization = None,
            show = False, fixed_params = [],
            **kwargs):
        if ax is None and self.ax is not None:
            ax = self.ax
        else:
            _, ax = plt.subplots()
        return plot_peak_progression(self.patterns, normalization = normalization, ax = ax,
            show = show, fixed_params = fixed_params, **kwargs)

#@utils.eager_persist_to_file("cache/xrd.get_normalization/")
def get_normalization(patterns, type = 'transmission', peak_width = config.peak_width,
        fixed_params = [], **kwargs):
    def get_max(pattern):
        # TODO: data types
        angles, intensities = map(lambda x: np.array(x), [pattern.angles, pattern.intensities])
        # TODO: is this necessary?
        return np.max(intensities[angles > 15.])
    if type == 'maximum':
        return np.array(map(get_max, patterns))
    if type == 'transmission':
        return np.array([pat.get_attribute('transmission') for pat in patterns])
    elif type == 'background':
        # TODO: reimplement this
        raise NotImplementedError
    elif type == 'peak': # Normalize by size of first peak
        return np.array([pat.peak_sizes(fixed_params = fixed_params)[1] for pat in patterns])
    else: # Interpret type as the name of a function in config.py
        try:
            return np.array([eval('config.%s' % type)(pat.label) for pat in patterns])
        except AttributeError:
            raise ValueError("Function config.%s(<image array>) not found." % type)

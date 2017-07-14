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
import query
from dataccess.utils import extrap1d
from scipy.interpolate import interp1d

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

def get_flux_density(pat):
    #TODO: make this function take a dataref. move config.py functions into another file
    beam_energy = config.beam_intensity_diagnostic(pat)
    size = pat.get_attribute('focal_size')
    # convert length units from microns to cm
    return beam_energy / (np.pi * ((size * 0.5 * 1e-4)**2))

def peak_progression(patterns, normalization = None,
        peak_width = config.peak_width, fixed_params = [],
        estimation_method = 'integral_bgsubbed', **kwargs):
    """
    Note: this function may only be called if the elements of labels are
    google spreadsheet dataset labels, rather than paths to data files.

    estimation_method : str 
        value == 'fit', 'integral', or 'integral_bgsubbed'. Determines the
        scheme for peak intensity estimation.
    """
    if normalization is None:
        normalization = 'transmission'

    compound_name = _patterns_compound(patterns)
    # sort by increasing beam intensity


    patterns = sorted(patterns, key = get_flux_density)
    # TODO: fix peak progression calculation for two-detid mode
    powder_angles = np.array(geometry.get_powder_angles(compound_name))

    label_flux_densities = map(get_flux_density, patterns)

    # indices: label, peak
    peaksize_array = np.array([pat.peak_sizes(method = estimation_method) for pat in patterns])
    normalized_peaksize_array = peaksize_array / get_normalization(patterns,
        peak_width = peak_width, normalization_type = normalization, fixed_params = fixed_params,
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
            fixed_params = fixed_params, **kwargs)

    output = []
    # TODO: reimplement selection of most intense peaks
    def plotting(ax):
        if ax is None:
            f, ax = plt.subplots(1)
        if logscale:
            if type(ax) == list:
                ax = ax[0]
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
def process_dataset(dataset, nbins = 2000, fiducial_ellipses = None,
        bgsub = True, **kwargs):
    pat = Pattern.from_dataset(dataset, nbins = nbins, fiducial_ellipses = fiducial_ellipses,
            bgsub = bgsub, **kwargs)
    return pat.angles, pat.intensities

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
    def __init__(self, dataref, detid, compound_list, model = None, mask = True,
            label = '', event_mask = None, **kwargs):
        """
        kwargs are frame-processing related arguments, including
        event_data_getter and frame_processor. See data_access.py for details.
        """
        self.dataref = dataref
        self.detid = detid
        self.compound_list = compound_list
        self.mask = mask
        self.event_mask = event_mask
        self.kwargs = kwargs
        if type(dataref) == np.ndarray:
            self.label = label
#            if label is None:
#                raise ValueError("label must be provided for dataset of type numpy.ndarray")
        else: # dataref must be a 
            self.label = dataref.label

    def get_array(self, event_data_getter = None, **kwargs):
        from dataccess import data_access as data

        #elif dataset.ref_type == 'label':
        if type(self.dataref) == np.ndarray:
            imarray, event_data =  self.dataref, None

        else:# Assume this is a query.DataSet instance
            imarray, event_data = data.eval_dataset_and_filter(self.dataref, self.detid,
                event_mask = self.event_mask, **self.kwargs)
        imarray = imarray.T

        if self.mask:
            base_mask = (imarray != 0.)
            extra_masks = config.detinfo_map[self.detid].extra_masks
            combined_mask = utils.combine_masks(base_mask, extra_masks, transpose = True)
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

# TODO: multiple models
def background_model(angle):
    from lmfit import Model
    def poly(x, a = 0, b = 0, c = 0, d = 1):
        return c * (x - angle) + d
        #return a * (x - angle)**3 + b * (x - angle)**2 + c * (x - angle) + d
    #from lmfit.models import PolynomialModel
    #return PolynomialModel(order)
    return Model(poly)

# Returns a second-order polynomial expanded about the center of the fit range
# TODO: linear/quadratic fit options
def poly_model():
    from lmfit import Model
    def shifted_poly(x, center = 0, slope = 0, intercept = 200, amplitude = 10):
        x = x - np.mean(x)
        return intercept + slope * (x - center)# - amplitude * (x - center)**2
    mod = Model(shifted_poly) 
    return mod

def model_fit(x, y, model, x_filter = lambda x: True, recenter = False,
        fixed_params = [], param_values = {}, **kwargs):
    from copy import deepcopy
    x, y = np.array(x), np.array(y)
    i_fit = np.where(x_filter(x))[0]
    i_sample = np.where(np.logical_and(x >= x[i_fit[0]], x < x[i_fit[-1]]))[0]
    pars = model.make_params(**param_values)
    for name, p in pars.iteritems():
        if p.name in fixed_params:
            p.set(vary = False)
    result = model.fit(y[i_fit], pars, x=x[i_fit])
    result.xfit = x[i_sample]
    eval_params = deepcopy(result.best_values)
    eval_params['x'] = x[i_sample]
    result.yfit = model.eval(**eval_params)
    return result

def eval_fitresult(model, result, x):
    from copy import deepcopy
    eval_params = deepcopy(result.best_values)
    eval_params['x'] = x
    return model.eval(**eval_params)

# TODO: why doesn't this class store powder data?
class Peak:
    """
    Class representing a single powder peak.
    """
    def __init__(self, angle, param_values, fixed_params, model = None,
            peak_width = config.peak_width, background_width = config.peak_width/2.):
        """
        angle : float
            Angle of the powder peak.
        param_values : {str: float}
            map from parameter names to starting values 
        fixed_params : list of str
            list of parameters to be fixed to their starting values
        model : lmfit.Model
            model to which to fit the peak. The model's parameters are assumed
            to be a subset of 'amplitude', 'center', 'sigma', 'slope',
            'intercept', and 'fraction' (the last being specific to lmfit's
            pseudo-Voigt model).
        """
        default_model_params = {'amplitude': 10, 'center': angle, 'sigma': .1,
                'slope': 0., 'intercept': 10., 'fraction': 0.5}
        #if model is 
        self.angle = angle
        self.param_values = {k: param_values.get(k, default_model_params[k]) for k in default_model_params.keys()}
        self.fixed_params = {k: (k in fixed_params and fixed_params[k]) for k in default_model_params.keys()}
        if set(param_values.keys()) != set(fixed_params):
            raise ValueError("Cannot constrain parameters without providing starting values.")
        self.peak_width = peak_width
        self.background_width = background_width
        self.model = model if model is not None else Peak._default_model()

    def _is_fixed(self, param):
        return (param in self.fixed_params and self.fixed_params[param])

    @staticmethod
    def _default_model():
        from lmfit import Model
        def gaussian(x, amplitude, center, sigma):
            "1-d gaussian: gaussian(x, amplitude, center, sigma)"
            return (amplitude/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x-center)**2 /(2*sigma**2))
        #def line(x, slope, intercept):
        def line(x, slope, intercept):
            "line"
            return slope * x + intercept
        mod = Model(gaussian) + Model(line)
        return mod

    # TODO: move this functionality into a custom model class
    def _get_model_params(self, mod, x = None, y = None):
        param_names = mod.param_names
        result = {k: v for k, v in self.param_values.iteritems() if k in param_names}
#        if x is not None and y is not None and 'center' in param_names:
#            assert type(y) == np.ndarray
#            assert type(x) == np.ndarray
#            result['center'] = x[np.argmax(y)]
        return result

    def _post_fit_update_params(self, best_values):
        self.param_values = {k: v for k, v in best_values.iteritems()}
#        for p in self.param_values:
#            self.param_values[p] = best_values[p] if p in self.param_values

    def set_model(self, model):
        self.model = model

    def integrate(self, x, y, mode = 'integral', fixed_params = [], param_values = {}):
        """
        TODO: make this more sophisticated.
        """
        x = np.array(x)
        i = self.crop_indices(x, self.peak_width)
        raw_integral = np.sum(np.array(y)[i])
        if mode == 'integral':
            return raw_integral
        elif mode == 'integral_bgsubbed':
            model = background_model(self.angle)
            fit_result = self.background_fit(x, y, model, fixed_params = fixed_params,
                    param_values = param_values)
            return raw_integral - np.sum(eval_fitresult(model, fit_result, x[i]))

#    def center_of_mass(self, x, y, **kwargs):
#        """
#        TODO: make this more sophisticated.
#        """
#        x = np.array(x)
#        i = self.crop_indices(x, self.peak_width)
#        x_i = x[i]
#        y_i = np.array(y)[i]
#        return np.sum(x_i * y_i) / np.sum(y_i)

    def center_of_mass(self, x, y, mode = 'mean', **kwargs):
        log('center of mas mode: %s' % mode)
        fitresult = self.peak_fit(x, y, **kwargs)
        if mode == 'mean': # calculate CM from weighted mean of data
            x = np.array(x)
            i = self.crop_indices(x, self.peak_width)
            x_i = x[i]
            y_i = np.array(y)[i]
            result = np.sum(x_i * y_i) / np.sum(y_i)
            log(np.shape(result))
            return result
        elif mode == 'fit': # calculate from peak fit
            result = fitresult.values['center']
            #log(np.shape(result))
            return result
        else:
            raise ValueError('invalid mode: %s' % str(mode))

    def background_fit(self, x, y, model, recenter = False, fixed_params = [],
            param_values = {}, **kwargs):
        """
        Fit an lmfit model to a segment of the provided pattern that's centered on this peak, 
        but excludes the peak region itself.
        """
        def x_filter(x):
            # define endpoints of the two background regions, on either side of the peak,
            # to which we will fit.
            lower_start, lower_end = self.angle - self.peak_width/2. - self.background_width, self.angle - self.peak_width/2.
            upper_start, upper_end = self.angle + self.peak_width/2., self.angle + self.peak_width/2. + self.background_width
            return ((lower_start <= x) & (x < lower_end)) | ((upper_start <= x) & (x < upper_end))
            #return np.logical_or(lower_start <= x < lower_end, upper_start <= x < upper_end)
        return model_fit(x, y, model, x_filter = x_filter, recenter = recenter,
                fixed_params = fixed_params, param_values = param_values, **kwargs)

    # TODO: refactor using model_fit
    def peak_fit(self, x, y, peak_width = config.peak_width, recenter = False, **kwargs):
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
        FitResult = namedtuple('Fit', ['amplitude', 'xfit', 'yfit', 'values'])

        i = self.crop_indices(x, self.peak_width, y = y, recenter = recenter)
        pars  = self.model.make_params(**self._get_model_params(self.model, x = x[i], y = y[i]))

        for name, p in pars.iteritems():
            # TODO: refactor
            if self._is_fixed(p.name):# or p.name == 'center':
                p.set(vary = False)

        result = self.model.fit(y[i], pars, x=x[i])
        self.result = result
        best_values = result.best_values
        # TODO!!!!
        if 'a' in best_values:
            best_values['amplitude'] = best_values['a']
        # The fit values are sometimes inverted, for some reason
        if 'sigma' in best_values and best_values['amplitude'] < 0 and best_values['sigma'] < 0:
            best_values['amplitude'] = -best_values['amplitude']
            best_values['sigma'] = -best_values['sigma']
        self._post_fit_update_params(best_values)
        xfit, yfit = x[i], result.best_fit

        amplitude = np.abs(best_values['amplitude'])
        #m, b = best_values['slope'], best_values['intercept']
        return FitResult(amplitude, xfit.copy(), yfit.copy(), best_values)

    def crop_indices(self, x, width = None, y = None, recenter = False):
        """ Return powder pattern indices centered around the given angle """
        if width is None:
            width = self.peak_width
        peakmin = lambda: self.angle - width/2.
        peakmax = lambda: self.angle + width/2.
        indices = lambda: np.where(np.logical_and(x >=peakmin(), x <= peakmax()))[0]
        # crop indices based on current value of self.angles
        nominal = indices()
        if y is not None and recenter:
            assert type(y) == np.ndarray
            self.angle = x[nominal][np.argmax(y[nominal])]
            return indices()
        return nominal

class PeakParams:
    """
    Store a set powder peaks, each consisting of an angle value and of
    peak-fitting parameters.

    Methods:
        fit_peaks : return fit parameters for all peaks, using a guassian +
        linear background model.
    """
    def __init__(self, angles, starting_values = None, fixed_params = None, model = None,
            peak_widths = None):
        """
        starting_values : list of dicts of form {str -> float}
            Starting values (or, for constrained parameters, final values) for
            one or more of the fit parameters 'sigma', 'center', and 'amp' for the
            gaussian components of peak fits. The default mapping is: {'amp':
            5, 'center': 5, 'sigma': 2}.
        fixed_params : list of tuples of strings
            Specifies which parameters to fix using the values specified in
            starting_values.
        """
        peaks = []
        if angles:
            if starting_values is None:
                starting_values = [{} for _ in angles]
            if fixed_params is None:
                fixed_params = [[] for _ in angles]
            for angle, d, fixed in zip(angles, starting_values, fixed_params):
                peaks.append(Peak(angle, d, fixed, model = model))
#        for angle in angles:
#            peaks.append(Peak(angle, starting_values, fixed_params))

        self.peaks = peaks

    def _get_valid_peaks_widths(self, x, y, peak_widths = None):
        if peak_widths is None:
            peak_widths = len(self.peaks) * [config.peak_width]
        # (peak, width) tuples
        return [(peak, peak_width)
                for peak, peak_width in zip(self.peaks, peak_widths)
                if (peak.angle - peak_width/2. >= np.min(x)) and
                        (peak.angle + peak_width/2. <= np.max(x))]

    def fit_peaks(self, x, y, peak_widths = None, **kwargs):
        if 'peak_width' in kwargs:
            del kwargs['peak_width']
        valid_peaks_widths = self._get_valid_peaks_widths(x, y, peak_widths = peak_widths)
        return [peak.peak_fit(x, y, peak_width = peak_width, **kwargs) for peak, peak_width
                in valid_peaks_widths]

    # TODO: refactor!!!!@!!
    def fit_backgrounds(self, x, y, peak_widths = None, **kwargs):
        valid_peaks_widths = self._get_valid_peaks_widths(x, y, peak_widths = peak_widths)
        # TODO:  take background model as a parameter
        return [peak.background_fit(x, y, background_model(peak.angle), peak_width = peak_width, **kwargs) for peak, peak_width in valid_peaks_widths]

    def __add__(self, other):
        import copy
        new = copy.deepcopy(self)
        new.peaks = new.peaks + copy.deepcopy(other.peaks)
        return new


class Pattern:
    def __init__(self, angles, intensities, compound_list, dataset = None,
            image  = None, peak_width = config.peak_width, label = '',
            nbins = 1000, fiducial_ellipses = None, bgsub = False,
            starting_values = None, model = None, pattern_smoothing = 0.,
            peak_angles = None, fixed_params = None, **kwargs):
        """
        Initialize an instance using explicit data
        
        kwargs are passed to Dataset.get_array()
        """
        self.width = peak_width
        self.compound_list = compound_list
        self.label = label

        self.angles = np.array(angles)
        self.intensities = gaussian_filter(intensities, pattern_smoothing)
        self.image = image
        self.dataset = dataset
        #self.angles = np.array(self.angles)
        self.anglemask = RealMask((np.min(self.angles), np.max(self.angles)))
        if peak_angles is None and self.compound_list is not None:
            self.peak_angles = np.array(geometry.get_powder_angles(self.get_compound(),
                filterfunc = lambda ang: self.anglemask.includes(ang)))
        elif peak_angles:
            self.peak_angles = peak_angles
        else:
            self.peak_angles = None
        self.peaks = PeakParams(self.peak_angles, starting_values = starting_values,
                fixed_params = fixed_params, model = model)


    def get_pattern(self):
        return self.angle, self.intensities

    @classmethod
    def from_xrdset(cls, xrdset, nbins = 1000, fiducial_ellipses = None,
            bgsub = False, pre_integration_smoothing = 0, **kwargs):
        """
        Instantiate from an XRDset instance.
        """
        if type(xrdset.dataref) == query.DataSet:
            dataset = xrdset.dataref
        else:
            dataset = None
        if 'label' in kwargs and kwargs['label'] is None:
            kwargs['label'] = xrdset.label
        angles, intensities, image =\
                geometry.process_imarray(xrdset.detid, xrdset.get_array(kwargs),
                        compound_list = xrdset.compound_list, nbins = nbins,
                        fiducial_ellipses = fiducial_ellipses, bgsub = bgsub,
                        pre_integration_smoothing = pre_integration_smoothing)
        return cls(angles, intensities, xrdset.compound_list, image =image,
                **kwargs)


    #@utils.eager_persist_to_file("cache/xrd/pattern.from_dataset/")
    @classmethod
    def from_dataset(cls, dataset, detid, compound_list, label = None, **kwargs):
        """
        Instantiate using a 2d data array from an area detector.

        kwargs are passed to Pattern.__init__().
        """
        xrdset = XRDset(dataset, detid, compound_list, label = label)
        return Pattern.from_xrdset(xrdset, label = label, dataset = dataset, **kwargs)

    @classmethod
    def from_event_patterns(cls, dataset, frame_processor, peak_width = config.peak_width,
             **kwargs):
        """
        This requires a frame_processor function that returns a powder pattern.
        """
        result = dataset.evaluate(frame_processor = frame_processor,
            event_data_getter = utils.identity, **kwargs)
        angles, intensities = np.sum(result.flat_event_data()).T
        return cls(angles, intensities, peak_width = peak_width)
        

    def get_attribute(self, attr):
        """
        Return an attribute value for the dataset(s) associated with this instance.
        """
        if self.dataset is not None:
            return self.dataset.get_attribute(attr)
        else:
            raise KeyError("Cannot get attribute from None dataset")

    @classmethod
    def from_multiple(cls, dataset_list, **kwargs):
        """
        Construct an instance from multiple Dataset instances.
        """
        assert len(dataset_list) >= 1
        components = [Pattern.from_dataset(ds, **kwargs) for ds in dataset_list]
        return reduce(operator.add, components)

    def __add__(self, other):
        """
        Add the data of a second Pattern instance to the current one.
        """
        assert self.width == other.width
        assert self.compound_list == other.compound_list
        self.angles = np.concatenate((self.angles, other.angles))
        self.intensities = np.concatenate((self.intensities, other.intensities))
        self.anglemask = self.anglemask + other.anglemask
        self.peak_angles = np.concatenate((self.peak_angles, other.peak_angles))
        self.peaks = self.peaks + other.peaks
        if self.image is not None and other.image is not None:
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
        if not self.compound_list:
            return None
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

    def fit_backgrounds(self, **kwargs):
        return self.peaks.fit_backgrounds(self.angles, self.intensities, **kwargs)

    # TODO: make this model-dependent
    @utils.ifplot
    def plot_peakfits(self, ax = None, show = False, normalization = 1.,
            peak_width = config.peak_width, bg = True, fit = True, plot_cm = False):
        # TODO: leverage lmfit better
        if ax is None:
            ax = self._new_ax()
        for peak_fit, bg_fit, cm in zip(self.fit_peaks(), self.fit_backgrounds(),
            self.centers_of_mass()):
            # TODO: fix this hack. how do we prevent specifying the color kwarg from screwing up plotly's color cycle?
            #ax.plot(fit.xfit, (fit.yfit - (fit.m * fit.xfit + fit.b)) / normalization, color = 'black')
            if bg:
                ax.plot(bg_fit.xfit, bg_fit.yfit / normalization, color = 'black', label = '')
            if fit:
                ax.plot(peak_fit.xfit, peak_fit.yfit / normalization, color = 'red')
            if plot_cm:
                ax.plot([cm, cm], [0, np.max(self.intensities)], color = 'green')
        if show:
            plt.show()
        return ax

    @utils.ifplot
    def plot_bgfits(self, ax = None, show = False, normalization = 1.,
            peak_width = config.peak_width):
        # TODO: leverage lmfit better
        if ax is None:
            ax = self._new_ax()
        for fit in self.fit_backgrounds():
            # TODO: fix this hack. how do we prevent specifying the color kwarg from screwing up plotly's color cycle?
            #ax.plot(fit.xfit, (fit.yfit - (fit.m * fit.xfit + fit.b)) / normalization, color = 'black')
            ax.plot(fit.xfit, fit.yfit / normalization, color = 'black')
            ax.plot(fit.xfit, fit.yfit / normalization, color = 'red')
        if show:
            plt.show()
        return ax

    @playback.db_insert
    @utils.ifplot
    def plot(self, ax = None, label = None, show = False, normalization = None,
            peak_width = config.peak_width, fixed_params = None, legend = False):
        if ax is None:
            ax = self._new_ax()
        if normalization:
            scale = get_normalization([ self ], normalization_type = normalization,
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
        elif normalization == 'integral_bgsubbed':
            self.plot_bgfits(ax = ax, show = False, peak_width = peak_width,
                    normalization = scale)
        if legend:
            plt.legend()
        if show:
            plt.show()
        return ax, scale

    def peak_sizes(self, peak_width = config.peak_width, fixed_params = None,
            method = 'fit', background_model  = None, background_fixed_params = [],
            background_param_values = {}, **kwargs):
        amplitudes = []
        # TODO: take the fitting function as a parameter for the 'fit' option
        if method == 'fit':
            peak_fits = self.peaks.fit_peaks(self.angles, self.intensities, peak_width = peak_width, fixed_params = fixed_params)
            for fit in peak_fits:
                amplitudes.append(fit.amplitude)
            return np.array(amplitudes)
        elif method == 'integral':
            return [peak.integrate(self.angles, self.intensities, mode = method) for peak in self.peaks.peaks]
        elif method == 'integral_bgsubbed':
            return [peak.integrate(self.angles, self.intensities, mode = method,
                fixed_params = background_fixed_params, param_values = background_param_values)
                    for peak in self.peaks.peaks]
            
        else:
            raise ValueError("invalid method %s" % method)

    def subtract_background(self, bg_pattern = None, bg_subtract = True,
        photon_value = 100., scale_factors_bg = None, **kwargs):
        """
        Return new, background-subtracted Pattern instance
        """
        angles, intensities = self.angles, self.intensities
        if bg_subtract:
            if bg_pattern is not None:
                peak_bgs = bg_pattern.fit_backgrounds()
            else:
                log( 'fitting background')
                peak_bgs = self.fit_backgrounds()

            if scale_factors_bg is None:
                scale_factors_bg = [1.] * len(peak_bgs)
            for scale, fit in zip(scale_factors_bg, peak_bgs):
                intensities = utils.interpolated_subtraction(angles, intensities, fit.xfit, scale * fit.yfit)
            # TODO clone all attributes
            return Pattern(angles, intensities, self.compound_list)
                    
    def centers_of_mass(self, bg_pattern = None, bg_subtract = True, mode = 'mean',
        photon_value = 100., scale_factors_bg = None, **kwargs):
        """
        Calculate peak centers of mass
        
        bg_pattern: a Pattern instance to be used for background subtraction
        via call to fit_backgrounds().

	if bg_pattern is None and bg_subtract is True, then a background fit to
	this Pattern instance will be used.
        """
        #pdb.set_trace()
        angles, intensities = self.angles, self.intensities
        if bg_subtract:
            if bg_pattern is not None:
                peak_bgs = bg_pattern.fit_backgrounds()
            else:
                log( 'fitting background')
                peak_bgs = self.fit_backgrounds()

            if scale_factors_bg is None:
                scale_factors_bg = [1.] * len(peak_bgs)
            for scale, fit in zip(scale_factors_bg, peak_bgs):
                # calculate CM with a regularization term
#                num_counts_bg = np.sum(fit.yfit) / photon_value
#                scale_factor_bg = (1. - 1./np.sqrt(num_counts_bg))
                log(scale)
                intensities = utils.interpolated_subtraction(angles, intensities, fit.xfit, scale * fit.yfit)
        return [peak.center_of_mass(angles, intensities, mode = mode)
                    for peak in self.peaks.peaks]

    # TODO: refactor
    def recentered_peaks(self, shift_scale = -1., corrections = None, **kwargs):
        """
         If shift values are not provided, the peaks are shifted so that their
         CMs correspond to nominal peak angles.
        """
        assert type(self.angles) == np.ndarray
        assert type(self.intensities) == np.ndarray
        peak_cms = self.centers_of_mass(**kwargs)
        output = self.intensities.copy()
        for peak, cm in zip(self.peaks.peaks, peak_cms):
            shift = shift_scale * (peak.angle - cm)
            i = ((peak.angle  - 0.5 *  peak.peak_width) < self.angles) & ((peak.angle  + 0.5 *  peak.peak_width) > self.angles)
            x_peak, y_peak = self.angles[i], self.intensities[i]
            _, new_y = utils.regrid(x_peak, y_peak, shift = shift)
            if not np.isnan(np.sum(new_y)):
                output[i] = new_y
        return self.angles, output

    def shift_peaks(self, corrections, shift_scale = -1.):
        """
        Return powder pattern with peaks shifted by the provided values.
        """
        output = self.intensities.copy()
        for peak, shift in zip(self.peaks.peaks, corrections):
            shift = shift_scale * shift
            i = ((peak.angle  - 0.5 *  peak.peak_width) < self.angles) & ((peak.angle  + 0.5 *  peak.peak_width) > self.angles)
            x_peak, y_peak = self.angles[i], self.intensities[i]
            _, new_y = utils.regrid(x_peak, y_peak, shift = shift)
            if not np.isnan(np.sum(new_y)):
                output[i] = new_y
        return Pattern(self.angles, output, self.compound_list)



    def background(self, peak_width = config.peak_width):
        """
        Return an estimate of background level by integrating from 0 to just
        before the first peak.
        """
        max_angle = self.peaks.peaks[0].angle - peak_width/2.
        x = np.array(self.angles)
        i =  np.where(x < max_angle)[0]
        return np.sum(np.array(self.intensities)[i])
                



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
        fixed_params = None, fit = True, event_masks = None, **kwargs):

        self.fixed_params = fixed_params

        if bgsub:
            if not compound_list:
                bgsub = False
                log( "No compounds provided: disabling background subtraction.")

        if event_masks is not None:
            assert len(event_masks) == len(data_identifiers)
        else:
            event_masks = [None] * len(data_identifiers)

        def pattern_one_dataref(dataref, event_mask = None):
            def one_detid(detid):
                dataset = XRDset(dataref, detid, compound_list, mask = mask,
                        event_data_getter = event_data_getter, frame_processor = frame_processor,
                        event_mask = event_mask, **kwargs)
                return Pattern.from_xrdset(dataset, label = dataset.label, bgsub = bgsub, starting_values = starting_values,
                        fixed_params = fixed_params, **kwargs)
            return reduce(operator.add, map(one_detid, detid_list))

        self.patterns =\
                [pattern_one_dataref(dataref, event_mask = event_mask)
                for dataref, event_mask in zip(data_identifiers, event_masks)]

        if not ax:
            self.ax = None
        else:
            self.ax = ax

        if fit:
            self.fit_all()

    def fit_all(self):
        for pat in self.patterns:
            pat.fit_peaks()

    def map_peak(self, peak_index, func):
        """
        For each powder pattern, call the provided function func (Peak -> Peak)
        on the Peak instance with index peak_index and replace that Peak
        instance with the return value.
        
        Returns a list of Peaks.
        """
        result = []
        for pat in self.patterns:
            peaks = pat.peaks.peaks
            peaks[peak_index] = func(peaks[peak_index])
            result.append(peaks[peak_index])
        return result

    def set_model_param(self, param, value, peak_index, fixed = True):
        """
        Set initial value of a peak fit parameter, optionally constraining it.
        
        param : str, equal to 'sigma', 'slope', or 'amplitude'
            parameter to set
        value : float
            parameter value
        peak_index : int
            index of the peak for which apply the change.
        """
        for p in self.patterns:
            peaks = p.peaks.peaks
            peaks[peak_index].param_values[param] = value
            if fixed:
                peaks[peak_index].fixed_params[param] = True

    def iter_peaks(self):
        """
        Iterate through all Peak instances contained in this dataset's Powder instances.
        """
        for pattern in self.patterns:
            for peak in pattern.peaks.peaks:
                yield peak

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
def get_normalization(patterns, normalization_type = 'transmission', peak_width = config.peak_width,
        fixed_params = [], background_model = None, **kwargs):
    #pdb.set_trace()
    def get_max(pattern):
        # TODO: data types
        angles, intensities = map(lambda x: np.array(x), [pattern.angles, pattern.intensities])
        # TODO: is this necessary?
        return np.max(intensities[angles > 15.])

    def eval_dataset(pattern, function):
        ds = pattern.dataset
        if ds is None:
            raise ValueError("Cannot normalize by io on Pattern instance with non-intitialized dataset attribute")
        # TODO: move this into a user-modifiable file
        return function(ds)

    if normalization_type == 'maximum':
        return np.array(map(get_max, patterns))
    if normalization_type == 'transmission':
        return np.array([pat.get_attribute('transmission') for pat in patterns])
    if normalization_type == 'i0':
        return np.array([config.beam_intensity_diagnostic(pat) for pat in patterns])
    elif normalization_type == 'background':
        raise NotImplementedError

    # TODO: document this feature
    elif callable(normalization_type):
        return np.array([eval_dataset(pat, normalization_type) for pat in patterns])

    elif normalization_type == 'peak': # Normalize by size of first peak
        return np.array([pat.peak_sizes(fixed_params = fixed_params)[1] for pat in patterns])
    elif normalization_type == 'integral_bgsubbed':
        return np.array(
                [pat.peak_sizes(fixed_params = fixed_params, method = normalization_type,
                        background_model = background_model)[1]
                    for pat in patterns]
                )
    else: # Interpret type as the name of a function in config.py
        try:
            return np.array([eval('config.%s' % normalization_type)(pat.label) for pat in patterns])
        except AttributeError:
            raise ValueError("Function config.%s(<image array>) not found." % normalization_type)

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

def _peak_size_gaussian_fit(x, y, amp = 5, cen = 5, wid = 2):
    from lmfit import Model
    def gaussian(x, amp, cen, wid):
        "1-d gaussian: gaussian(x, amp, cen, wid)"
        return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))

    def line(x, slope, intercept):
        "line"
        return slope * x + intercept

    mod = Model(gaussian) + Model(line)
    pars  = mod.make_params( amp=amp, cen=cen, wid=wid, slope=0, intercept=1)

    result = mod.fit(y, pars, x=x)
    return result.best_values, x, result.best_fit

def _patterns_compound(patterns):
    compound_list = [pattern.get_compound() for pattern in patterns]
    if not _all_equal(compound_list):
        raise ValueError("pattern compounds missing, or mismatching")
    return compound_list.pop()


    
def plot_peak_progression(patterns, maxpeaks = 'all', ax = None, logscale = True,
        normalization = None, show = False, inner_normalization = True, **kwargs):

    powder_angles, pattern_fluxes, progression, normalized_progression =\
            peak_progression(patterns, normalization = normalization)

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


def peak_progression(patterns, normalization = None,
        peak_width = config.peak_width, **kwargs):
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
        peak_width = peak_width, type = normalization, **kwargs)[:, np.newaxis]

    # indices: peak, label
    heating_progression = normalized_peaksize_array.T
    normalized_heating_progression = heating_progression / heating_progression[:, 0][:, np.newaxis]
    return powder_angles, label_flux_densities, heating_progression, normalized_heating_progression


# TODO: why does memoization fail?
#@utils.eager_persist_to_file("cache/xrd.process_dataset/")
def process_dataset(dataset, nbins = 1000, fiducial_ellipses = None,
        bgsub = True, **kwargs):
    def process_one_detid(detid):
        imarray = dataset.get_array(detid)#data_extractor(dataset, **kwargs)
        return geometry.process_imarray(detid, imarray,
            fiducial_ellipses = fiducial_ellipses, bgsub = bgsub,
            compound_list = dataset.compound_list, **kwargs)
    return reduce(operator.add, map(process_one_detid, dataset.detid_list))

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
    def doplot():
        if plot_progression:
            xrd.plot_progression(ax = ax2, maxpeaks = maxpeaks)
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
        """
        # TODO: update this
        Returns CSPAD image data in the correct format for all other functions in
        this module given a data path, run group label, or raw array. Optionally
        masks out pixels using mask files specified in config.extra_masks.

        Parameters
        ---------
        path : str
            path to an ASCII datafile
        label : str
            run group label specified from the logbook
        arr : numpy.ndarray
            2d array of CSPAD data
        """
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


class Pattern:
    def __init__(self, xrdset, peak_width = config.peak_width,
            nbins = 1000, fiducial_ellipses = None, bgsub = True,
            pre_integration_smoothing = 0, **kwargs):
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

    @utils.ifplot
    def plot_peakfits(self, ax = None, show = False, normalization = 1.,
            peak_width = config.peak_width):
        if ax is None:
            ax = self._new_ax()
        for m, b, amplitude, xfit, yfit in self.peak_fits(peak_width = peak_width):
            ax.plot(xfit, yfit / normalization, color = 'red')
            ax.plot(xfit, (yfit - (m * xfit + b)) / normalization, color = 'black')
        if show:
            plt.show()
        return ax

    @playback.db_insert
    @utils.ifplot
    def plot(self, ax = None, label = None, show = False, normalization = None,
            peak_width = config.peak_width):
        if ax is None:
            ax = self._new_ax()
        if normalization:
            scale = get_normalization([ self ], type = normalization, ax = ax)[0]
        else:
            scale = 1.
        if label is None:
            label = self.label
        dthet = (np.max(self.angles) - np.min(self.angles))/len(self.angles)
        ax.plot(self.angles, gaussian_filter(self.intensities, 0.05/dthet)/scale, label = label)
        label_angles = np.array(geometry.get_powder_angles(self.get_compound(),
            filterfunc = lambda ang: self.anglemask.includes(ang)))
        if label_angles is not None:
            for ang in label_angles:
                ax.plot([ang, ang], [np.min(self.intensities)/scale, np.max(self.intensities)/scale], color = 'black')
        #pdb.set_trace()
        if normalization == 'peak':
            self.plot_peakfits(ax = ax, show = False, peak_width = peak_width,
                    normalization = scale)
        plt.legend()
        if show:
            plt.show()
        return ax

    def peak_fits(self, peak_width = config.peak_width, **kwargs):
        """
        Returns a list of tuples of the form (m, b, amplitude, xfit, yfit), where:
            m: linear coeffient of background fit
            b: offset of background fit
            amplitude: amplitude of fit's gaussian component
            xfit : np.ndarray. x-values of the fit range
            yfit : np.ndarray. fit evaluated on the array xfit
        """
        x, y = map(np.array, [self.angles, self.intensities])
        def fit_one_peak(peakmin, peakmax):
            dx = np.mean(np.abs(np.diff(x)))
            i = np.where(np.logical_and(x >=peakmin, x <= peakmax))[0]
            amp_approx = np.sum(y[i] - np.min(y[i])) * dx
            center = (peakmin + peakmax) / 2.
            best_values, xfit, yfit = _peak_size_gaussian_fit(x[i], y[i], amp = amp_approx, cen = center, wid = 0.2)
            # The fit values are sometimes inverted, for some reason
            if best_values['amp'] < 0 and best_values['wid'] < 0:
                best_values['amp'] = -best_values['amp']
                best_values['wid'] = -best_values['wid']
            amplitude = best_values['amp']
            m, b = best_values['slope'], best_values['intercept']
            return m, b, amplitude, xfit.copy(), yfit.copy()

        def in_range(angle_degrees):
            return angle_degrees > np.min(x) and angle_degrees < np.max(x)
        fitlist = []
        powder_angles = np.array(geometry.get_powder_angles(self.get_compound(), filterfunc = in_range))
        
        make_interval = lambda angle: [angle - peak_width/2.0, angle + peak_width/2.0]
        make_ranges = lambda angles: map(make_interval, angles)

        # ranges over which to integrate the powder patterns
        peak_ranges = make_ranges(powder_angles)

        for peakmin, peakmax in peak_ranges:
            peakIndices = np.where(np.logical_and(x >= peakmin, x <= peakmax))[0]
            if len(peakIndices) == 0:
                raise ValueError("Peak angle %s: outside data range.")
            fitlist += [fit_one_peak(peakmin, peakmax)]
        return fitlist

    def peak_sizes(self, peak_width = config.peak_width, ax = None, **kwargs):
        peak_fits = self.peak_fits(peak_width = peak_width)
        amplitudes = []
        for m, b, amplitude, xfit, yfit in peak_fits:
            amplitudes.append(amplitude)
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
        plot_progression = False, plot_peakfits = False, show_image = True,
        event_data_getter = None, frame_processor = None, **kwargs):

        if bgsub:
            if not compound_list:
                bgsub = False
                log( "No compounds provided: disabling background subtraction.")

        def pattern_one_dataref(dataref):
            def one_detid(detid):
                dataset = XRDset(dataref, detid, compound_list, mask = mask,
                        event_data_getter = event_data_getter, frame_processor = frame_processor,
                        **kwargs)
                return Pattern(dataset, bgsub = bgsub)
            #pdb.set_trace()
            return reduce(operator.add, map(one_detid, detid_list))

        #pdb.set_trace()
        self.patterns = map(pattern_one_dataref, data_identifiers)

        if plot_peakfits and not ax:
            _, self.ax = plt.subplots()
        elif not ax:
            self.ax = None
        else:
            self.ax = ax
        
    def show_images(self):
        imarrays = [pat.images for pat in self.patterns]
        labels = [pat.label for pat in self.patterns]
        for im, lab in zip(imarrays, labels):
            utils.show_image(im, title = lab)

    def normalize(self, mode):
        # TODO
        raise NotImplementedError

    def plot_patterns(self, ax = None, normalization = None, **kwargs):
        if ax is None and self.ax is not None:
            ax = self.ax
        else:
            ax = None
        for pat in self.patterns:
            pat.plot(ax = ax, normalization = normalization)

    def plot_progression(self, ax = None, maxpeaks = 6, normalization = None,
            show = False, **kwargs):
        if ax is None and self.ax is not None:
            ax = self.ax
        else:
            _, ax = plt.subplots()
        return plot_peak_progression(self.patterns, normalization = normalization, ax = ax,
            show = show, **kwargs)
    
#@utils.eager_persist_to_file("cache/xrd.get_normalization/")
def get_normalization(patterns, type = 'transmission', peak_width = config.peak_width,
        ax = None, **kwargs):
    #pdb.set_trace()
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
        return np.array([pat.peak_sizes(ax = ax)[0] for pat in patterns])
    else: # Interpret type as the name of a function in config.py
        try:
            return np.array([eval('config.%s' % type)(pat.label) for pat in patterns])
        except AttributeError:
            raise ValueError("Function config.%s(<image array>) not found." % type)

"""
Functions for getting derived data from MEC-specific diagnostics.
"""
import data_access
import utils
import numpy as np
from dataccess import xes_process as spec

# based on equating GMD output to si integral for run 880
si_integral_to_eV = 7.1923263553224339e-11

def background_subtracted_spectrum(imarr, transpose = False):
    if transpose:
        imarr = imarr.T
    cencol = spec.center_col(imarr)
    return spec.bgsubtract_linear_interpolation(spec.lineout(imarr, cencol, pxwidth = 30))

def si_imarr_sum(imarr, **kwargs):
    spectrum = background_subtracted_spectrum(imarr, transpose = True)
    return np.sum(spectrum)

def si_imarr_cm_3(imarr, **kwargs):
    imarr = imarr.T
    x, y = spec.get_spectrum(imarr, calib_load_path = '/reg/neh/home5/ohoidn/analysis/MgO/si_calib')
    #spectrum = background_subtracted_spectrum(imarr, transpose = True)
    cm = np.sum(y * x)/np.sum(y)
    return cm
    #return (cm_index * e_scale) + e_intercept

def si_imarr_cm_4(imarr, **kwargs):
    x, y = spec.get_spectrum(imarr, calib_load_path = '/reg/neh/home5/ohoidn/analysis/MgO/si_calib')
    #spectrum = background_subtracted_spectrum(imarr, transpose = True)
    cm = np.sum(y * x)/np.sum(y)
    return cm
    #return (cm_index * e_scale) + e_intercept

def si_imarr_cm_subregion_3(imarr, **kwargs):
    imarr = imarr.T
    imarr = imarr[400:660, :]
    x, y = spec.get_spectrum(imarr)
    #spectrum = background_subtracted_spectrum(imarr, transpose = True)
    cm = np.sum(y * x)/np.sum(y)
    return cm
    #return (cm_index * e_scale) + e_intercept

#TODO: should have an mec-specific config file
# TODO: Add a generic function for the integral of a detector in xes_process.py
@utils.eager_persist_to_file('cache/mec/si_spectrometer_integral')
def si_spectrometer_integral(label, **kwargs):
    # Dark frame-subtracted si spectrometer data
    mean_frame, event_data = data_access.eval_dataset_and_filter(label, 'si', event_data_getter = si_imarr_sum)
    return np.sum(background_subtracted_spectrum(mean_frame))

@utils.eager_persist_to_file('cache/mec/xrts1_fe_fluorescence_integral')
def xrts1_fe_fluorescence_integral(label):
    def xrts1_sum(imarr, **kwargs):
        indices, spectrum = background_subtracted_spectrum(imarr)
        return np.sum(spectrum)
    mean_frame, _ = data_access.eval_dataset_and_filter(label, 'xrts1')
    return xrts1_sum(mean_frame)

def grid_mask(arr, stride = 10):
    mask = np.ones(arr.shape, dtype = bool)
    mask[::stride, :] = False
    mask[:, ::stride] = False
    return mask

def outlier_mask(arr, min = -10, max = 10):
    mask = np.ones(arr.shape, dtype = bool)
    mask[np.logical_or(arr < min, arr > max)] = False
    return mask

# TODO: handle the mec-specific data files referenced here
def write_masks():
    m1 = np.load('quad1_mask_calibman_10.12.npy')

    m2 = np.load('quad2_mask_calibman_10.12.npy')
    #m2 = np.load('mask3.npy')

    dark1 = np.load('dark1.npy').T
    new1 = outlier_mask(dark1) * m1
    #new1 = zero_islands(outlier_mask(dark1) * grid_mask(dark1) * m1, 0, 75)

    dark2 = np.load('dark2.npy').T
    new2 = outlier_mask(dark2) * m2
    #new2 = zero_islands(outlier_mask(dark2) * grid_mask(dark2) * m2, 0, 75)

    np.save('quad1_mask_10.11.npy', new1)

    np.save('quad2_mask_10.11.npy', new2)

"""
Functions for getting derived data from MEC-specific diagnostics.
"""
import data_access
import utils
import numpy as np

def background_subtracted_spectrum(imarr, transpose = False):
    from dataccess import xes_process as spec
    if transpose:
        imarr = imarr.T
    cencol = spec.center_col(imarr)
    return spec.bgsubtract_linear_interpolation(spec.lineout(imarr, cencol, pxwidth = 30))

#TODO: should have an mec-specific config file
# TODO: Add a generic function for the integral of a detector in xes_process.py
@utils.eager_persist_to_file('cache/mec/si_spectrometer_integral')
def si_spectrometer_integral(label):
    def si_imarr_sum(imarr, **kwargs):
        spectrum = background_subtracted_spectrum(imarr, transpose = True)
        return np.sum(spectrum)
    # Dark frame-subtracted si spectrometer data
    mean_frame, event_data = data_access.get_data_and_filter(label, 'si', event_data_getter = si_imarr_sum)
    return utils.dict_leaf_mean(event_data)

@utils.eager_persist_to_file('cache/mec/xrts1_fe_fluorescence_integral')
def xrts1_fe_fluorescence_integral(label):
    def xrts1_sum(imarr, **kwargs):
        spectrum = background_subtracted_spectrum(imarr)
        return np.sum(spectrum)
    mean_frame, _ = data_access.get_data_and_filter(label, 'xrts1')
    return xrts1_sum(mean_frame)
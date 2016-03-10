"""
Functions for getting derived data from MEC-specific diagnostics.
"""
import data_access
import utils
import numpy as np

def si_background_subtracted_spectrum(imarr):
    from dataccess import xes_process as spec
    imarr = imarr.T
    cencol = spec.center_col(imarr)
    return spec.bgsubtract_linear_interpolation(spec.lineout(imarr, cencol, pxwidth = 30))

#TODO: should have an mec-specific config file
@utils.eager_persist_to_file('cache/mec/si_spectrometer_integral')
def si_spectrometer_integral(label):
    def si_imarr_sum(imarr, **kwargs):
        spectrum = si_background_subtracted_spectrum(imarr)
        return np.sum(spectrum)
    # Dark frame-subtracted si spectrometer data
    mean_frame, event_data = data_access.get_data_and_filter(label, 'si', event_data_getter = si_imarr_sum)
    return utils.dict_leaf_mean(event_data)

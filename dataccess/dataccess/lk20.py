import numpy as np
from dataccess import xrd
from dataccess.peakfinder import peakfilter_frame

def powder_pattern_shift_cm5(imarr = None, **kwargs):
    pat = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
    #pdb.set_trace()
    return np.array(pat.recentered_peaks())
#def ith_peak_cm_bgsubbed(i, bg_pat):
#    def powder_pattern_cm(imarr = None, **kwargs):
#        from dataccess.peakfinder import peakfilter_frame
#        from dataccess import xrd
#        dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
#        
#        return dss.centers_of_mass(bg_pattern = bg_pat)[i]
#    return powder_pattern_cm

#def ith_peak_cms_bgsubbed(bg_pat):
#    def powder_pattern_cm(imarr = None, **kwargs):
#        dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr, detid = 'quad2'), 'quad2', ['MgO'], label  = 'test')
#        return np.array(dss.centers_of_mass(bg_pattern = bg_pat))
#    return powder_pattern_cm


def cm_variation_scatter_plot_bgsubbed(i, ds):
    from dataccess.summarymetrics import plt
    from dataccess.nbfunctions import eval_xrd
    from dataccess import utils
    from scipy.interpolate import interp1d
    from scipy.ndimage.filters import gaussian_filter as gfilt
    def func(ds):
        def ith_peak_cms_bgsubbed(bg_pat, i):
            def powder_pattern_cm(imarr = None, **kwargs):
                from dataccess import xrd
                from dataccess.peakfinder import peakfilter_frame
                dss = xrd.Pattern.from_dataset(peakfilter_frame(imarr,detid = 'quad2'),'quad2',['MgO'],label= 'test')
                return dss.centers_of_mass ( bg_pattern = bg_pat )[i]

            return powder_pattern_cm

        x = eval_xrd([ds], ['MgO'], normalization = 'integral_bgsubbed', detectors = ['quad2', 'quad1'], frame_processor = peakfilter_frame)
        bg_pat = x.patterns[0]
        trace = ds.evaluate('quad2', frame_processor = ith_peak_cms_bgsubbed(bg_pat, i), event_data_getter = utils.identity)
        event_numbers, cms = range(len(trace.flat_event_data())), trace.flat_event_data()
        smoothed_cm_interp = utils.extrap1d(interp1d(event_numbers, gfilt(cms, 20)))
        smoothed_cms = smoothed_cm_interp(event_numbers)
        cms_highpass = cms - smoothed_cms
        #from dataccess.mec import si_imarr_cm_3
        #energies = ds.evaluate('si', frame_processor = si_imarr_cm_3, event_data_getter = utils.identity)

        plt.scatter(event_numbers, cms, label = 'raw')
        plt.plot(np.array(event_numbers), smoothed_cms, label  ='interpolation, smoothed with sigma = 20')
        plt.show()
    return func(ds)

#cvsp0 = cm_variation_scatter_plot_bgsubbed(0)
#cvsp1 = cm_variation_scatter_plot_bgsubbed(1)

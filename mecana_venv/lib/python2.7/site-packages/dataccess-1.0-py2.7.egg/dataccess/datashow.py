from multiprocessing import Process
from dataccess import utils
from dataccess import data_access
import matplotlib.pyplot as plt

def apply_default_masks(imarray, detid):
    import config
    extra_masks = config.detinfo_map[detid].extra_masks
    combined_mask = utils.combine_masks(imarray, extra_masks)
    return imarray * combined_mask

def identity(imarr):
    return imarr


def one_plot(label, detid, path = None, masked = False, rmin = None, rmax = None, run = None, plot = True, show = True):
    if run is None:
        imarray, _ = data_access.get_data_and_filter(label, detid)
        if masked:
            imarray = apply_default_masks(imarray, detid)
        if not path:
            path = 'datashow_images/' + label + '_' + str(detid)
        if plot:
            utils.save_image_and_show(path, imarray, title = label + '_' + detid, rmin = rmin, rmax = rmax, show_plot = show)
    else:
        imarray, framesdict = data_access.get_label_data(label, detid, event_data_getter = identity)
        frames = framesdict.values()[0].values()
        if not path:
            path = 'datashow_images/' + label + '_' + str(detid)
        if plot:
            utils.save_image_and_show(path, frames[run], title = label + '_' + detid + '_run ' + str(run), rmin = rmin, rmax = rmax, show_plot = show)

def main(labels, detid, **kwargs):
    for label in labels:
        one_plot(label, detid, show = False, **kwargs)
    plt.show()

#def main(label, detid, path = None, masked = False, rmin = None, rmax = None, runs = None):
#    if run is None:
#        imarray, _ = data_access.get_data_and_filter(label, detid)
#        if masked:
#            imarray = apply_default_masks(imarray, detid)
#        if not path:
#            path = 'datashow_images/' + label + '_' + str(detid)
#        utils.save_image_and_show(path, imarray, title = label + '_' + detid, rmin = rmin, rmax = rmax)
#    else:
#        imarray, frames = data_access.get_label_data(label, detid, event_data_getter = identity)
#        for run in runs:
#            utils.save_image_and_show(path, frames[run], title = label + '_' + detid + '_run ' + str(run), rmin = rmin, rmax = rmax)

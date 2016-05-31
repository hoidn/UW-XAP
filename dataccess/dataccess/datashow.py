from multiprocessing import Process
import matplotlib.pyplot as plt

import utils
import data_access
import query

def apply_default_masks(imarray, detid):
    import config
    extra_masks = config.detinfo_map[detid].extra_masks
    combined_mask = utils.combine_masks(imarray, extra_masks)
    return imarray * combined_mask

def identity(imarr):
    return imarr


def one_plot(label, detid, path = None, masked = False, rmin = None, rmax = None, run = None, plot = True, show = True):
    dataset = query.existing_dataset_by_label(label)
    if run is None:
        imarray, _ = data_access.eval_dataset_and_filter(dataset, detid)
        if masked:
            imarray = apply_default_masks(imarray, detid)
        if not path:
            path = 'datashow_images/' + label + '_' + str(detid)
        if plot:
            utils.save_image_and_show(path, imarray, title = label + '_' + detid, rmin = rmin, rmax = rmax, show_plot = show)
    else:
        imarray, framesdict = data_access.eval_dataset(dataset, detid, event_data_getter = identity)
        frames = framesdict.values()[0].values()
        if not path:
            path = 'datashow_images/' + label + '_' + str(detid)
        if plot:
            utils.save_image_and_show(path, frames[run], title = label + '_' + detid + '_run ' + str(run), rmin = rmin, rmax = rmax, show_plot = show)

def main(labels, detid, **kwargs):
    for label in labels:
        one_plot(label, detid, show = False, **kwargs)
    plt.show()


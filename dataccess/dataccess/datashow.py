from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np

import data_access
import query
import xrd

def apply_default_masks(imarray, detid):
    import config
    import utils
    extra_masks = config.detinfo_map[detid].extra_masks
    combined_mask = utils.combine_masks(imarray, extra_masks)
    return imarray * combined_mask

def identity(imarr):
    return imarr


def one_plot(label, detid, path = None, masked = False, rmin = None, rmax = None, run = None, plot = True, show = True, fiducials = []):
    import utils
    def put_fiducials(imarray):
        (phi, x0, y0, alpha, r) = xrd.get_detid_parameters(detid)
        betas, rho = xrd.get_beta_rho(imarray, phi, x0, y0, alpha, r)
        new = imarray.copy()
        for angle in fiducials:
            new = np.where(np.logical_and(betas < angle + .1, betas > angle - .1),
                0., new)
        return new
    dataset = query.existing_dataset_by_label(label)
    if run is None:
        imarray, _ = data_access.eval_dataset_and_filter(dataset, detid)
        imarray = imarray.T
        if masked:
            imarray = apply_default_masks(imarray, detid)
        if not path:
            path = 'datashow_images/' + label + '_' + str(detid)
        if plot:
            utils.save_image_and_show(path, put_fiducials(imarray), title = label + '_' + detid, rmin = rmin, rmax = rmax, show_plot = show)
    else:
        imarray, framesdict = data_access.eval_dataset(dataset, detid, event_data_getter = identity)
        imarray = imarray.T
        frames = framesdict.values()[0].values()
        if not path:
            path = 'datashow_images/' + label + '_' + str(detid)
        if plot:
            utils.save_image_and_show(path, put_fiducials(frames[run]), title = label + '_' + detid + '_run ' + str(run), rmin = rmin, rmax = rmax, show_plot = show)

def main(labels, detid, fiducials = [], **kwargs):
    for label in labels:
        one_plot(label, detid, show = False, fiducials = fiducials, **kwargs)
    plt.show()


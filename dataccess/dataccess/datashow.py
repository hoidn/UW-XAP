from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np
import pdb

import data_access
import query
import geometry

import config

if config.plotting_mode == 'notebook':
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('matplotlib inline')

def apply_default_masks(imarray, detid):
    import config
    import utils
    extra_masks = config.detinfo_map[detid].extra_masks
    combined_mask = utils.combine_masks(imarray, extra_masks)
    return imarray * combined_mask

def identity(imarr):
    return imarr


def one_plot(dataset, detid, path = None, masked = False, rmin = None, rmax = None, run = None,
        plot = True, show = True, fiducials = [], darksub = True, frame_processor = None):
    import utils
    def put_fiducials(imarray):
        try:
            (phi, x0, y0, alpha, r) = geometry.get_detid_parameters(detid)
        except KeyError:
            raise KeyError("%s: geometry configuration parameter phi not found.")
        betas, rho = geometry.get_beta_rho(imarray, phi, x0, y0, alpha, r)
        new = imarray.copy()
        for angle in fiducials:
            new = np.where(np.logical_and(betas < angle + .1, betas > angle - .1),
                0., new)
        return new
    if run is None:
        imarray, _ = data_access.eval_dataset_and_filter(dataset, detid, darksub = darksub,
                frame_processor = frame_processor)
        imarray = imarray.T
        if masked:
            imarray = apply_default_masks(imarray, detid)
        if not path:
            path = 'datashow_images/' + dataset.label + '_' + str(detid)
        if fiducials:
            annotated = put_fiducials(imarray)
        else:
            annotated = imarray
        if plot:
            utils.save_image_and_show(path, annotated, title = dataset.label + '_' + detid, rmin = rmin, rmax = rmax, show_plot = show)
    else:
        imarray, framesdict = data_access.eval_dataset(dataset, detid,
                event_data_getter = identity, frame_processor = frame_processor)
        imarray = imarray.T
        frames = framesdict.values()[0].values()
        if not path:
            path = 'datashow_images/' + dataset.label + '_' + str(detid)
        annotated = put_fiducials(frames[run])
        if plot:
            utils.save_image_and_show(path, annotated, title = dataset.label + '_' + detid + '_run ' + str(run), rmin = rmin, rmax = rmax, show_plot = show)
    return annotated

def show(datasets, detid, fiducials = [], **kwargs):
    result =\
        [one_plot(dataset, detid, show = True, fiducials = fiducials, **kwargs)
        for dataset in datasets]
    return result

def main(dataref_list, detid, fiducials = [], **kwargs):
    """
    dataset: a list of labels or query.DataSet instances.
    """

    def to_dataset(ref):
        if isinstance(ref, str):
            return query.existing_dataset_by_label(ref)
        else:
            return ref

    return show(map(to_dataset, dataref_list), detid, fiducials = fiducials, **kwargs)


import matplotlib.pyplot as plt
from dataccess import utils
from dataccess import data_access

def apply_default_masks(imarray, detid):
    import config
    extra_masks = config.detinfo_map[detid].extra_masks
    combined_mask = utils.combine_masks(imarray, extra_masks)
    return imarray * combined_mask

def main(label, detid, path = None, masked = False, rmin = 0, rmax = 3000):
    imarray, _ = data_access.get_data_and_filter(label, detid)
    if masked:
        imarray = apply_default_masks(imarray, detid)
    if not path:
        path = 'datashow_images/' + label + '_' + str(detid)
    utils.save_image_and_show(path, imarray, title = label + '_' + detid, rmin = rmin, rmax = rmax)

import os
from dataccess import data_access
from scipy import misc


def main(label, detid, filtered = False):
    save_dir = 'eventdata/' + str(detid) + '_' + label + '/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p ' + os.path.dirname(save_dir))
    def process_event(imarr, nevent = None, **kwargs):
#        if imarr.dtype == 'uint16':
#            imarr = imarr.astype('float')
        misc.imsave(save_dir + str(nevent) + '.jpg', imarr)

    if not filtered:
        dispatch = data_access.get_label_data
    else:
        dispatch = data_access.get_data_and_filter
    dispatch(label, detid, event_data_getter = process_event)

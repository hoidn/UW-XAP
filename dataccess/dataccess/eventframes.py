from scipy import misc
import os

import data_access
import query


def main(label, detid, filtered = False):
    dataset = query.existing_dataset_by_label(label)
    save_dir = 'eventdata/' + str(detid) + '_' + label + '/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p ' + os.path.dirname(save_dir))
    def process_event(imarr, nevent = None, **kwargs):
#        if imarr.dtype == 'uint16':
#            imarr = imarr.astype('float')
        misc.imsave(save_dir + str(nevent) + '.jpg', imarr)

    if not filtered:
        dispatch = data_access.eval_dataset
    else:
        dispatch = data_access.eval_dataset_and_filter
    dispatch(dataset, detid, event_data_getter = process_event)

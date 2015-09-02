import os
import avg_bgsubtract_hdf
import data_access

def main():
    all_clusters = avg_bgsubtract_hdf.get_run_clusters()
    for cluster in all_clusters:
        if len(cluster) > 2:
            os.system('python src/avg_bgsubtract_hdf.py ' + ' '.join(map(str, cluster)))
#    for label in data_access.get_label_map().keys():
#        try:
#            data_access.get_label_data(label, 1)
#            data_access.get_label_data(label, 2)
#        except:
#            print label, ' failed'

#if __name__ == '__main__':
#    main()

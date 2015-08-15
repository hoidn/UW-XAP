import os
import avg_bgsubtract_hdf

def main():
    all_clusters = avg_bgsubtract_hdf.get_run_clusters()
    for cluster in all_clusters:
        if len(cluster) > 5:
            os.system('python src/avg_bgsubtract_hdf.py ' + ' '.join(map(str, cluster)))

#if __name__ == '__main__':
#    main()

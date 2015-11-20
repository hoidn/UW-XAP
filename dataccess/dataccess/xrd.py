# Valenza, Ditter, and Hoidn
# Based on bojangles, by A. Ditter and R. Valenza


#import ConfigParser
import os
import numpy as np
import matplotlib.pyplot as plt

import config
from dataccess import data_access as data

global verbose
verbose = True

## Dict of position parameters
#config_data = {}
#def xrd_init(detid):
#    """
#    Initialize position parameters for a given detector
#    """
#    tocopy = config.xrd_config[detid]
#    for k, v in tocopy.iteritems():
#        config_data[k] = v

# ConfigParser-related stuff: no longer necessary after consolidating configs
# under config.py
#config_name = 'detector_coords.cfg'
#config_handle = 'coords'
#def make_empty_config():
#    """
#    Create an empty configuration file
#    """
#    config = ConfigParser.ConfigParser()
#    with  open(config_name, 'w') as cfg_file:
#        config.add_section(config_handle)
#        config.set(config_handle, 'phi')
#        config.set(config_handle, 'x0')
#        config.set(config_handle, 'y0')
#        config.set(config_handle, 'alpha')
#        config.set(config_handle, 'r')
#        config.write(cfg_file)
#
#def config_exists():
#    return os.path.exists(config_name)
#
#    
#def load_config():
#    config = ConfigParser.ConfigParser()
#    if not os.path.exists(config_name):
#        make_empty_config()
#        raise IOError("Config file not found; created empty one")
#    else:
#        config.read(config_name)
#
#    def map_config(section):
#        options = config.options(section)
#        for option in options:
#            try:
#                config_data[option] = config.getfloat(section, option)
#                if config_data[option] == -1:
#                    print "skip: %s" % option
#            except:
#                print "exception on %s!" % option
#    map_config(config_handle)
#
#def config_loaded():
#    for k, v in config_data.iteritems():
#        if not v:
#            return False
#    return True

def data_extractor(path = None, label = None, detid = None, run_label_filename = None):
    if all([label, detid]):
        return data.get_label_data(label, detid, fname = run_label_filename)
    elif all([path, detid]):
        return np.genfromtxt(path)
    else:
        raise ValueError("Invalid argument combination. Data source must be specified by detid and either path or label and run_label_filename")


# translate(phi, x0, y0, alpha, r)
# Produces I vs theta values for imarray. For older versions, see bojangles_old.py
# Inputs:  detector configuration parameters and diffraction image
# Outputs:  lists of intensity and 2theta values (data)
def translate(phi, x0, y0, alpha, r, imarray):
    length, width = imarray.shape
    y = np.vstack(np.ones(width)*i for i in range(length))
    ecks = np.vstack([1 for i in range(length)])
    x = np.hstack(ecks*i for i in range(width))
    x2 = -np.cos(phi) *(x-x0) + np.sin(phi) * (y-y0)
    y2 = -np.sin(phi) * (x-x0) - np.cos(phi) * (y-y0)
    rho = (r**2 + x2**2 + y2**2)**0.5
    y1 = y2 * np.cos(alpha) + r * np.sin(alpha)
    z1 = - y2 * np.sin(alpha) + r * np.cos(alpha)
    # beta is the twotheta value for a given (x,y)
    beta = np.arctan2((y1**2 + x2**2)**0.5, z1) * 180 / np.pi
    imarray = imarray * np.square(rho)
    
    newpoints = np.vstack((beta.flatten(), imarray.flatten()))
    
    return newpoints.T, imarray


# binData()
# Input:  a minimum, a maximum, and a stepsize
# Output:  a list of bins

def binData(mi, ma, stepsize, valenza = True):
    
    if verbose: print "creating angle bins"
    binangles = list()
    binangles.append(mi)
    i = mi
    while i < ma-(stepsize/2):
        i += stepsize
        binangles.append(i)

    return binangles


# processData()
# Inputs:  a single diffraction image array
# Outputs:  data in bins, intensity vs. theta. Saves data to file
def processData(imarray, paramdict, nbins = 1000, verbose = True):
    """
    TODO: need docstring!!!
    """

    imarray = imarray.T
    # Manually entered data after 2015/04/01 calibration. (really)
    # See Testing.nb for details.
    # Coordinates in pixels. 0.011cm per pixel.
    # LD67 inputs
    #(phi, x0, y0, alpha, r) = (0.027763, 322.267, 524.473, 0.787745, 1082.1)
    (phi, x0, y0, alpha, r) = paramdict['phi'], paramdict['x0'],\
        paramdict['y0'], paramdict['alpha'], paramdict['r']
    data, imarray = translate(phi, x0, y0, alpha, r, imarray)
    
    thetas = data[:,0]
    intens = data[:,1]

    # algorithm for binning the data
    ma = max(thetas)
    mi = min(thetas)
    stepsize = (ma - mi)/(nbins)
    binangles = binData(mi, ma, stepsize)
    numPix = [0] * (nbins+1)
    intenValue = [0] * (nbins+1)
    
    if verbose: print "putting data in bins"        
    # find which bin each theta lies in and add it to count
    for j,theta in enumerate(thetas):
        if intens[j] != 0:
            k = int(np.floor((theta-mi)/stepsize))
            numPix[k]=numPix[k]+1
            intenValue[k]=intenValue[k]+intens[j]
    # form average by dividing total intensity by the number of pixels
    if verbose: print "adjusting intensity"
    adjInten = (np.array(intenValue)/np.array(numPix))
    
    return binangles, adjInten, imarray


def save_data(angles, intensities, save_path):
    dirname = os.path.dirname(save_path)
    if dirname and (not os.path.exists(dirname)):
        os.system('mkdir -p ' + os.path.dirname(save_path))
    np.savetxt(save_path, [angles, intensities])


def plot_patterns(patterns):
    for angles, intensities in patterns:
        plt.plot(angles, intensities)
    plt.show()


def main(detid, data_identifiers, run_label_filename = 'labels.txt', mode = 'labels',
plot = True):
    patterns = []
    paramdict = config.xrd_config[detid]
    for data_ref in data_identifiers:
        if mode == 'labels':
            imarray = data_extractor(label = data_ref, detid = detid, run_label_filename = run_label_filename)
        elif mode == 'paths':
            imarray = data_extractor(path = data_ref, detid = detid,
                run_label_filename = run_label_filename)
        binangles, adjInten, imarray = processData(imarray, paramdict)
        path = 'xrd_patterns/' + data_ref + '_' + str(detid)
        save_data(binangles, adjInten, path)
        patterns.append([binangles, adjInten])

    if plot:
        plot_patterns(patterns)
    return patterns

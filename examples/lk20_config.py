from collections import namedtuple
import numpy as np

# if True, use MPI with the new (psana V4?) smd interface to access data. Otherwise use
# the old (V3?) API.
smd = True

# Experiment specification. Example (for LD67):
# exppath = 'mec/mecd6714'
# xtc_prefix = "e441"
# This must be provided to run any analysis. 
exppath = "mec/meck2016"
try:
    expname = exppath.split('/')[1]
except:
    raise ValueError("config.exppath: incorrect format")

xtc_prefix = 'e691'

# url of the google doc logbook
url = "https://docs.google.com/spreadsheets/d/1cmSQysWXF6jBObNgnxqN6NJgRm_SzDT338b-mBj5kyc/edit#gid=0"
urls = None

use_logbook = True

# Probe photon energy in eV
photon_energy = 8910.
# Energy per pulse, in J
pulse_energy = 1e-4


# structure to store area detector information
# TODO: interfacing with Testing.nb.
# Quad CSPAD position parameters in the form of a dictionary of detector IDs
# to parameter dictionaries. Parameter values are obtain by running Alex's
# Mathematica notebook for this.  Coordinates are in pixels; 0.011cm per
# pixel.  See Testing.nb for details.  This information must be provided to
# run XRD analysis.
# Map from detector IDs to a list of 0 or more paths for additional mask files
# (beyond what psana applies to 'calibrated' frames). 
#   -For composite detectors, this program expects masks corresponding to
#   assembeled images.
#   -Multiple masks are ANDed together.
#   -Mask files must be boolean arrays saved in .npy format.
#   -Masks must be positive (i.e., bad/dummy pixels are False).
#DetInfo = namedtuple('DetInfo', ['device_name', 'dimensions', 'geometry', 'extra_masks', 'subregion_index'])
DetInfo = namedtuple('DetInfo', ['device_name', 'geometry', 'extra_masks', 'subregion_index'])

# Updated 1/23/2016. Calibrated with Fe3O4 data (runs 632 to 636)
#{'maskmaker/81_quad2mask_2_2.mask.npy'},
# -Alex
detinfo_map =\
    {'quad1':
        DetInfo('MecTargetChamber.0:Cspad.0',
        {'phi': 0, 'x0': 409, 'y0': 187, 'alpha': 2.33874, 'r': 1041.4},
        {'maskmaker/81_quad1mask_2_2.mask.npy'},
        0),
    'quad2':
        DetInfo('MecTargetChamber.0:Cspad.0',
        {'phi': -1.5708-0.0191543, 'x0': 819-164.835, 'y0': 574.192, 'alpha': 0.759423, 'r': 970.13},
        {'maskmaker/mask3.npy'},
        1),
    'allquads':
        DetInfo('MecTargetChamber.0:Cspad.0',
        {'phi': None, 'x0': None, 'y0': None, 'alpha': None, 'r': None},
        {},
        -1),
     'xrts1':
        DetInfo('MecTargetChamber.0:Cspad2x2.1',
        {},
        {},
        -1),
    'xrts2':
        DetInfo('MecTargetChamber.0:Cspad2x2.2', 
        {},
        {},
        -1),
    'vuv':
        DetInfo('MecTargetChamber.0:Princeton.1', 
        {},
        {},
        -1),
    'si':
        DetInfo('MecTargetChamber.0:Opal1000.1', 
        {},
        {},
        -1)}

## Map from BLD non-area detector ids to their full psana source names
#nonarea =\
#    {'d1': {'type': 'Lusi.IpmFexV1', 'src': 'MEC-TCTR-DI-01'},
#    'ipm2': {'type': 'Lusi.IpmFexV1', 'src': 'MEC-XT2-IPM-02'},
#    'ipm3': {'type': 'Lusi.IpmFexV1', 'src': 'MEC-XT2-IPM-03'},
#    'energy': {'type': 'Bld.BldDataFEEGasDetEnergyV1', 'src': 'FEEGasDetEnergy'}}

NonareaInfo = namedtuple('NonareaInfo', ['type', 'src'])
nonarea =\
    {'d1': NonareaInfo(
        'Lusi.IpmFexV1',
        'MEC-TCTR-DI-01'),
    'ipm2': NonareaInfo(
        'Lusi.IpmFexV1',
        'MEC-XT2-IPM-02'),
    'ipm3': NonareaInfo(
        'Lusi.IpmFexV1',
        'MEC-XT2-IPM-03'),
    'GMD': NonareaInfo( # beam energy gas detectors
        'Bld.BldDataFEEGasDetEnergyV1',
        'FEEGasDetEnergy')
    }

powder_angles = {'Fe3O4': [27.2, 32.1, 33.47, 38.9, 48.1, 51.3, 56.2], 'MgO': [0, 33.47, 38.9],
    'Graphite': [24.0, 30.2, 38.4, 41.6]}
#powder_angles = {'Fe3O4': [42.2, 50.5, 61.7, 74.4, 78.2]}

def si_spectrometer_probe(imarr):
    spectrum = np.sum(imarr, axis = 1)
    spectrum = spectrum - np.percentile(spectrum, 1) # subtract background
    probe_counts = np.sum(spectrum[330:403])
    return probe_counts

def si_spectrometer_pump(imarr):
    spectrum = np.sum(imarr, axis = 1)
    spectrum = spectrum - np.percentile(spectrum, 1) # subtract background
    pump_counts = np.sum(spectrum[403:520])
    return pump_counts

def make_si_filter(probe_min, probe_max, pump_min, pump_max):
    def filter_by_si_peaks(imarr):
        spectrum = np.sum(imarr, axis = 1)
        spectrum = spectrum - np.percentile(spectrum, 1) # subtract background
        probe_counts = np.sum(spectrum[330:403])
        pump_counts = np.sum(spectrum[403:520])
        if (probe_min < probe_counts < probe_max) and (pump_min < pump_counts < pump_max):
            return True
        return False
    return filter_by_si_peaks
    #return pump_counts, probe_counts

def sum_si(imarr):
    baseline = np.percentile(imarr, 1)
    return np.sum(imarr - baseline)

def identity(imarr):
    return imarr

def flux(beam_energy, label = None, size = None, **kwargs):
    from dataccess import data_access as data
    size = data.get_label_property(label, 'focal_size')
    flux = beam_energy * data.get_label_property(label, 'transmission') /  (np.pi * ((size * 0.5 * 1e-4)**2))
    return flux
    
## global filter detid
#filter_detid = 'xrts1'
## global event filter function
#def filter_function(imarr):
#    return filter_by_si_peaks(imarr)
#    #return True

# TODO (maybe): parameters for XES script

# port for ZMQ sockets
#port = "5558"

# Size in microns of the beam spot at best focus
best_focus_size = 2.

def sum_window(smin, smax):
    return lambda arr: smin < np.sum(arr) < smax

port = 5679

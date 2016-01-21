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
urls =\
    ["https://docs.google.com/spreadsheets/d/1Ejz-kyf-eoCtI1A76f1qfpS2m6nThosWDrh2dXUjv5g/edit?pref=2&pli=1#gid=0",
    "https://docs.google.com/spreadsheets/d/1rsyUi1GshHkc56bYGr0pIOiiSStXM1RHnEVAclF7qb0/edit#gid=0"]

# Probe photon energy in eV
photon_energy = None
# Energy per pulse, in J
pulse_energy = None

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

detinfo_map =\
    {'quad1':
        DetInfo('MecTargetChamber.0:Cspad.0',
        {'phi': None, 'x0': None, 'y0': None, 'alpha': None, 'r': None},
        {},
        0),
    'quad2':
        DetInfo('MecTargetChamber.0:Cspad.0',
        {'phi': None, 'x0': None, 'y0': None, 'alpha': None, 'r': None},
        {},
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

# Map from BLD non-area detector ids to their full psana source names
nonarea =\
    {'d1': 'MEC-TCTR-DI-01',
    'ipm2': 'MEC-XT2-IPM-02',
    'ipm3': 'MEC-XT2-IPM-03'}



# Size in microns of the beam spot at best focus
best_focus_size = 2.

def sum_window(smin, smax):
    return lambda arr: smin < np.sum(arr) < smax

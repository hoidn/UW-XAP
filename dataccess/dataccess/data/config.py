# TODO: figure out a detector ID naming convention based on not-too-verbose
# names from the psana framework

# Experiment specification. Example (for LD67):
#exppath = 'mec/mecd6714'
# This must be provided to run any analysis. 
exppath = None

# Quad CSPAD position parameters in the form of a dictionary of detector IDs
# to parameter dictionaries. Parameter values are obtain by running Alex's
# Mathematica notebook for this.  Coordinates are in pixels; 0.011cm per
# pixel.  See Testing.nb for details.  This information must be provided to
# run XRD analysis.
# Example (from LD67. The quad CSPAD's identifier is 3):
#xrd_geometry = {
#    3: {'phi': 0.027763, 'x0': 322.267, 'y0': 524.473, 'alpha': 0.787745, 'r': 1082.1}
#}

xrd_geometry = {
    3: {'phi': None, 'x0': None, 'y0': None, 'alpha': None, 'r': None}
}

# Map from detector IDs to a list of 0 or more paths for additional mask files
# (beyond what psana applies to 'calibrated' frames). 
#   -For composite detectors, this program expects masks corresponding to
#   assembeled images.
#   -Multiple masks are ANDed together.
#   -Mask files must be boolean arrays saved in .npy format.
#   -Masks must be positive (i.e., bad/dummy pixels are False).
extra_masks = {
    3: []
}

# Map from sample composition designator to either (1) a list of powder peak
# angles (units of degrees) or (2) a path to a file containing a simulated
# powder pattern from which this information can be extracted. This
# data is used for applying background subtraction in XRD analysis.
# For example:
#powder_angles = {
#    'Fe3O4': [31.3, 37.0, 45.1, 52.4, 56.0, 59.7, 65.7]
#}
powder_angles = {
}

# Identifier for google spreadsheet from which to download label data
# (optional)
# TODO: implement this
spreadsheet_url = None
sheet_indices = []

# TODO (maybe): parameters for XES script

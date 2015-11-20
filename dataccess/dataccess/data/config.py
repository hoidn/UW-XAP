# TODO: figure out a detector ID naming convention based on not-too-verbose
# names from the psana framework

# Experiment name specification. Example (from LD67):
#exppath = 'mec/mecd6714'
#expname = "mecd6714"

exppath = None
expname = None

# Quad CSPAD position parameters in the for of a dictionary of detector IDs
# to parameter dictionaries. Parameter values are obtain by running Alex's
# Mathematica notebook on XRD data.
# Coordinates in pixels. 0.011cm per pixel. See Testing.nb for 
# details. 
# Example (from LD67. The quad CSPAD's identifier is 3):
#xrd_config = {
#    3: {'phi': 0.027763, 'x0': 322.267, 'y0': 524.473, 'alpha': 0.787745, 'r': 1082.1}
#}
xrd_config = {
    3: {'phi': None, 'x0': None, 'y0': None, 'alpha': None, 'r': None}
}

# Identifier for google spreadsheet from which to download label data
# (optional)
# TODO: implement this
spreadsheet_url = None
sheet_indices = []

# TODO (maybe): parameters for xes analysis script

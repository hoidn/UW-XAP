# Author: O. Hoidn

import pandas as pd
import numpy as np
import utils

# row format:
# Ele.  A   Trans.  Theory (eV) Unc. (eV)   Direct (eV) Unc. (eV)   Blend  Ef
# Data source: http://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
with open(utils.resource_path('data/fluorescence.txt'), 'rb') as f:
    data = pd.read_csv(f, sep = '\t')

def emission_dict():
    """
    Returns a dict of dicts mapping element name and emission line label
    to photon energy in eV.

    The line label keys are: 'ka1', 'ka2', 'kb', and 'Ef' (Fermi energy)
    The element keys are 'Ne' through 'Fm'

    The energies used are from column 5 in fluorescence.txt. This data file
    currently contains complete data only for ka1, ka2, and kb. Ef energies
    for a few elements have been manually added.
    """
    line_dict = {}
    line_lookup = {'KL2': 'ka2', 'KL3': 'ka1', 'KM3': 'kb', 'Ef': 'Ef'}
    def process_one_row(row):
        name, line, energy = row[0], line_lookup[row[2]], row[5]
        elt_dict = line_dict.setdefault(name, {})
        elt_dict[line] = energy
    for i, row in data.iterrows():
        process_one_row(row)
    return line_dict


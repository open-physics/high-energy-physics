# https://fastjet.readthedocs.io/en/latest/Awkward.html

import fastjet
import awkward as ak


# Clustering specification
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.6)

# The data
array = ak.Array(
    [
        {"px": 1.2, "py": 3.2, "pz": 5.4, "E": 2.5, "ex": 0.78},
        {"px": 32.2, "py": 64.21, "pz": 543.34, "E": 24.12, "ex": 0.35},
        {"px": 32.45, "py": 63.21, "pz": 543.14, "E": 24.56, "ex": 0.0},
    ],
)

# ClusterSequence class
cluster = fastjet.ClusterSequence(array, jetdef)

# Extracting information
cluster.inclusive_jets()

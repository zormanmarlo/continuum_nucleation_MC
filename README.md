# Continuum Monte Carlo Model for Simulation of Nucleation

## Overview:
This code enables simulation of speciation and nucleation in a continuous coordinate space. Interactions are calculated based on a user-supplied set of tabulated potentials, which should be placed in the _potentials/_ directory. Repulsive potentials are switched from the infinitely dilute form to an screened form when two like ions are within the _clus_cutoff_ of the same unlike ion and within the _counter_cutoff_ of each other. As of right now, screened potentials are hardcoded in the _utils.py_ file and have been optimized to reproduce NaCl radial distribution functions. If you would like to turn off screened potentials, set the _counter_cutoff_ to zero.

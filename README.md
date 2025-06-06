# Continuum Monte Carlo Model for Simulation of Nucleation

## Overview
This code enables simulation of speciation and nucleation in a continuous coordinate space. Interactions are calculated based on a user-supplied set of tabulated potentials, which should be placed in the potentials/ directory. Repulsive potentials are switched from the infinitely dilute form to a screened form as a function of cluster size when two like ions are within the clus_cutoff of the same unlike ion and within the counter_cutoff of each other. As of right now, screened potentials are hardcoded for the JC and Dang forcefields. 

## Use example
Information for a simulation run is contained in the config file. 
- See _configs/commented_example.txt_ for a well-commented example.
- See _100mM_submit.sh_ for an example of how one might submit this script to a cluster. If running locally I recommend using fewer processes.
- Run _python simulation.py -h_ for information on flags for the python script
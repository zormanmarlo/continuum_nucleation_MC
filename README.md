# Monte Carlo Model for Simulation of Ion Speciation

## Overview
A Monte Carlo code that simulates speciation and nucleation of LJ particles in continous space. 

 
## Files and Classes

- `simulation.py`: Main simulation driver class, handles I/O and parallelization
- `system.py`: Physics implementation and Monte Carlo engine, handles particle positions, types, clusters, etc
- `config.py`: Configuration file parser and validation

**utils.py**
- `PMF`: Potential of mean force class for tabulated interactions
  - Generates LJ potential from parameters in config file
- `Bias`: Umbrella sampling bias potential implementation
  - Supports harmonic and linear bias types
  - Handles adaptive bias updates for enhanced sampling
- Numba-compiled functions: `calc_energy_numba()`, `find_neighbors_numba()`, `interpolate_energy_numba()`

**moves.py**
- `Move`: Base class for Monte Carlo moves with statistics tracking
- `TranslationMove`: Random particle displacement within spherical constraint
- `SwapMove`: Random particle repositioning anywhere in box
- `InOutAVBMCMove`/`OutInAVBMCMove`: Aggregation-Volume-Bias Monte Carlo moves
- `NVTInOutMove`/`NVTOutInMove`: Nucleation moves

## adapUS example

See **configs/S6_512mono_adapUS.txt** for example config file to use for adapUS. After each iteration, the simulation compiles target cluster samples for all markov chains (these are output at the output interval set in the configuration file) and uses this data to update bias. You may need to play around with both length and output intervals in order to improve convergence. The maximum target size is set in the configuration with the max_target variable. Convergence is defined as all bins being within 10% of the average bin count. For each iteration in which there are no empty bins but we have not converged, the simulation is extended by 20%.

In general, the adapUS runs are able to find a reasonable bias within a few iterations of all the bins being filled. After this point, I recommend taking that bias and using it as the input for a new adapUS run with longer iterations and a larger output interval. Pl

The command to specify that a simulation should use adaptive US:

```bash
# Run adaptive US simulation with 10 markov chains
python simulation.py -config configs/S6_512monoa_dapUS.txt -jobname S6_512mono_adapUS -np 10 -adapUS
```
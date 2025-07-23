# Monte Carlo Model for Simulation of Ion Speciation

## Overview
A Monte Carlo code that simulates speciation and nucleation of ions in a continuous coordinate space. Ions are represented as spherical particles. Interactions are calculated based on a user-supplied set of tabulated potentials, which should be placed in the potentials/ directory.

Note that this framework begins to fail at high concentrations/large cluster sizes, as the singular set of potentials is likely to inadequately represent particle interactions in denser environments. 
 
## Files and Classes

- `simulation.py`: Main simulation driver class, handles I/O and parallelization
- `system.py`: Physics implementation and Monte Carlo engine, handles particle positions, types, clusters, etc
- `config.py`: Configuration file parser and validation

**utils.py**
- `PMF`: Potential of mean force class for tabulated interactions
  - Loads and interpolates energy data from files
  - Supports different interaction types (ion-ion, cation-anion)
- `Bias`: Umbrella sampling bias potential implementation
  - Supports harmonic and linear bias types
  - Handles adaptive bias updates for enhanced sampling
- Numba-compiled functions: `calc_energy_numba()`, `find_neighbors_numba()`, `interpolate_energy_numba()`

**moves.py**
- `Move`: Base class for Monte Carlo moves with statistics tracking
- `TranslationMove`: Random particle displacement within spherical constraint
- `SwapMove`: Random particle repositioning anywhere in box
- `InOutAVBMCMove`/`OutInAVBMCMove`: Aggregation-Volume-Bias Monte Carlo moves
- `NVTInOutMove`/`NVTOutInMove`: Nucleation moves with Rosenbluth sampling

## Simple example

Run a basic NaCl nucleation simulation:

```bash
# Run a short test simulation (single processor)
python simulation.py -config configs/test_config.txt -jobname test_run

# Run with multiple processors for better statistics
python simulation.py -np 4 -config configs/test_config.txt -jobname parallel_test

# Run with custom output directory
python simulation.py -config configs/test_config.txt -jobname my_sim -path ./results
```

**Output Files Generated:**
- `E-XX.log`: Energy vs time trajectories
- `traj-XX.xyz`: Particle coordinates
- `clusters-XX.out`: Cluster size distributions over time
- `target_cluster-XX.out`: Size of cluster around particle 0
- `stats-XX.log`: Monte Carlo move acceptance rates

## Umbrella Sampling Example

Umbrella sampling along the nuclei size can be performed by running a set of biased simulations with increasing bias centers. This approach samples rare nucleation events by applying harmonic bias potentials at different cluster sizes.

### Required Files (in `US_files/` directory)

**Core Scripts:**
- **`run_us.sh`** - Main US driver script that iterates over cluster size centers, generates edited config files and slurm submission scripts
- **`submit_template.sh`** - Template slurm submission file that gets edited by `run_us.sh`
- **`gen_US_structs.py`** - Script to generate initial cluster configurations for each US window
- **`clean_colvars.sh`** - Post-processing script to combine colvar files from independent trajectories

**Configuration Templates:**
- **`configs/nacl_us/100mM_nacl_template_large_random_JC.txt`** - Example template config file for 100 mM NaCl system with JC forcefield
  - Key parameters: `bias_center` and `input_path` (automatically edited for each window)

**Input Structures:**
- **`inputs/nacl_us/100mM_nacl_32mer_large_random_XX.xyz`** - Pre-generated cluster configurations
  - Each window needs multiple input files (numbered `_00`, `_01`, etc.) for independent simulations

### Workflow Steps

1. **Generate Initial Structures**
   ```bash
   python US_files/gen_US_structs.py
   ```
   - Creates clusters of size N with randomly placed bath atoms
   - May take time for higher concentrations - edit parameters as needed
   - Generates multiple input files per window for statistical averaging

2. **Prepare Configuration Template**
   - Create config file with desired system parameters
   - Set `bias_type = harmonic` 
   - Leave `bias_center` and `input_path` as placeholders (auto-edited by driver)

3. **Configure Cluster Submission**
   - Edit `submit_template.sh` for your cluster environment
   - Set appropriate resource requests and module loads

4. **Run Umbrella Sampling Windows**
   ```bash
   bash US_files/run_us.sh
   ```
   - **Important:** Use `-multi_inputs` flag when running simulations
   - This tells the code that each process gets its own input file (`input_XX.xyz`)

5. **Post-Process Results**
   ```bash
   bash US_files/clean_colvars.sh
   ```
   - Combines colvar files from independent trajectories per window
   - Discards first half of each trajectory by default

6. **Analyze Free Energy**
   - Run WHAM (Weighted Histogram Analysis Method) on combined colvar files
   - Extract potential of mean force along cluster size coordinate

### Configuration Example

```bash
# Template config for umbrella sampling
bias_type = harmonic
bias_center = PLACEHOLDER    # Auto-replaced with window center
bias_k = 0.5
input_path = PLACEHOLDER     # Auto-replaced with window input file

# Standard simulation parameters
prod_steps = 1000000
output_interval = 1000
translation_rate = 0.8
avbmc_rate = 0.2
```

## Adaptive Umbrella Sampling Example

Adaptive umbrella sampling automatically optimizes bias potentials during the simulation to achieve uniform sampling across cluster sizes.

### Usage

```bash
python simulation.py -config your_config.txt -jobname adaptive_run -adapUS
```

### Configuration Requirements

```bash
# Enable adaptive bias
bias_type = linear
max_target = 50              # Maximum cluster size to sample

# Initial simulation parameters  
prod_steps = 500000          # Steps per adaptive iteration
equil_steps = 100000
output_interval = 1000

# Move parameters
translation_rate = 0.8
avbmc_rate = 0.2
```

### How It Works

1. **Initial Phase**: Runs unbiased simulation to collect initial cluster size distribution
2. **Iterative Optimization**: 
   - Analyzes cluster size histogram from all parallel processes
   - Updates bias potential using adaptive algorithm
   - Runs another simulation cycle with updated bias
3. **Convergence**: Continues until cluster size distribution becomes sufficiently flat
4. **Output**: Final optimized bias potential and enhanced sampling trajectories

### Output Files

- **`histograms.out`** - Cluster size distributions from each iteration
- **`potentials.out`** - Evolution of bias potential during adaptation
- **`system.pkl`** - Pickled system state for restarting/analysis
- Standard output files (`E-XX.log`, `traj-XX.xyz`, etc.)

### Advantages

- **Automatic optimization** - No need to manually choose bias parameters
- **Efficient sampling** - Rapidly converges to optimal bias for uniform sampling  
- **Parallel execution** - Uses all processors to accelerate convergence
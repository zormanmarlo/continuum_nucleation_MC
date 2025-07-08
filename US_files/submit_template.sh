#!/bin/bash
#SBATCH --job-name=100mM_CENTERmer_US
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=10:00:00
#SBATCH --mem=10gb
# E-mail Notification, see man sbatch for options

## SBATCH --workdir=$SLURM_SUBMIT_DIR

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

#module load intel

module load gcc/13.2.0
source /gscratch/cheme/mzorman/03_misc/miniconda3/etc/profile.d/conda.sh
conda activate
python simulation.py -np 5 -jobname CENTERmer -config configs/nacl_us/100mM_nacl_CENTERmer_large_random_JC.txt -path 100mM_nacl_US_large_random_JC

exit 0

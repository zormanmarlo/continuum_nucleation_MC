#!/bin/bash
#SBATCH --job-name=50mM_nacl
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=100:00:00
#SBATCH --mem=25gb
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
python simulation.py -np 10 -jobname ../driver_jobs/50mM_dang_adapUS -config configs/50mM_adapUS.txt -adapUS 

exit 0

#!/bin/bash
#SBATCH --job-name=100mM_100mer_switch
#SBATCH --account=pfaendtner
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=130:00:00
#SBATCH --mem=75gb
# E-mail Notification, see man sbatch for options

## SBATCH --workdir=$SLURM_SUBMIT_DIR

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

#module load intel

module load foster/python/miniconda/3.8
python3 simulation.py -np 20 -jobname 100mM_100mer -config configs/100mM_nacl_config.txt

exit 0

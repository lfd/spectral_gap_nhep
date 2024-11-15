#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=fhth
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# number of cores we need
#SBATCH --ntasks=8
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=05:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=8000MB
#
# infos
#
# output path
#SBATCH --output="logs/slurm/slurm-%j-%x.out"
d
module load devel/python/3.11.7_intel_2021.4.0

# Space for two arguments: pipeline and params
~/spectral_gap_nhep/.venv/bin/python -m kedro run $1 $2

# Done
exit 0



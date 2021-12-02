#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# sbatch /src/experiments/harmonics/slurm.sh

# SLURM guide: https://supercloud.mit.edu/submitting-jobs

# Slurm sbatch options
#SBATCH -o slurm-logs/harmonics.log-%j
#SBATCH -c 16
#SBATCH --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda/2021b
conda activate scaling

# Run the script
python -m src.experiments.harmonics.run_experiment

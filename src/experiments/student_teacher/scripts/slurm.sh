#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/student_teacher/scripts/slurm.sh
#
# For details on how this script works, see the SLURM guide here:
# https://supercloud.mit.edu/submitting-jobs.
#
# Also see https://stackoverflow.com/a/44168719/1337463.

sbatch <<EOT
#!/bin/bash
# Slurm sbatch options
#SBATCH -o slurm-logs/student_teacher/log-%j
#SBATCH -c 16
#SBATCH --gres=gpu:volta:1

# Load conda environment.
# See https://github.com/conda/conda/issues/7980#issuecomment-441358406
# for details on why we do it this way.
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2021b/etc/profile.d/conda.sh
conda activate scaling

# Set up and launch proxy
source scripts/rand-mallory.sh
{ mallory -config \$TMP_MALLORY_CONFIG; } &

# Run experiment
{ python -m src.experiments.student_teacher.run_experiment $@; } &

# Wait till experiment finishes, then kill mallory
# See https://unix.stackexchange.com/a/231678/466333 for details.
wait -n
pkill -P \$\$
EOT

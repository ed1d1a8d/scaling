#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/slurm.sh
#
# For details on how this script works, see the SLURM guide here:
# https://supercloud.mit.edu/submitting-jobs.
#
# Also see https://stackoverflow.com/a/44168719/1337463.

sbatch <<EOT
#!/bin/bash
# Slurm sbatch options
#SBATCH -o slurm-logs/ax/log-%j
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# Load conda environment.
# See https://github.com/conda/conda/issues/7980#issuecomment-441358406
# for details on why we do it this way.
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2021b/etc/profile.d/conda.sh
conda activate scaling-v2

# Set up and launch proxy
source scripts/rand-mallory.sh
{ mallory -config \$TMP_MALLORY_CONFIG; } &

# Utility function that waits until n background jobs are left running
# First argument is n
function waitUntilNJobsRemain() {
  local n_jobs=\$(jobs -rp | wc -l)
  echo "Waiting on \$n_jobs background jobs until \$1 jobs left..."
  while [[ \$n_jobs -gt \$1 ]]; do
    n_jobs=\$(jobs -rp | wc -l)
    echo -n "."
    sleep 1
  done
  echo ""
}

# Run experiment
n_trains=(2 4 10 20 50 80 100 200 300 500 1000 2000 5000 10000 20000 50000)
for n_train in "\${n_trains[@]}"
do
  LD_LIBRARY_PATH=/home/gridsan/groups/ccg/envs/scaling-v2/lib python -m src.ax.train --n_train \$n_train $@;
done

# Wait until all experiments finish.
waitUntilNJobsRemain 1

# Then kill mallory
pkill -P \$\$
EOT

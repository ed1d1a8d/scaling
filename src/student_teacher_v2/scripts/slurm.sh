#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./src/student_teacher_v2/scripts/slurm.sh
#
# For details on how this script works, see the SLURM guide here:
# https://supercloud.mit.edu/submitting-jobs.
#
# Also see https://stackoverflow.com/a/44168719/1337463.

sbatch <<EOT
#!/bin/bash
# Slurm sbatch options
#SBATCH -o slurm-logs/student-teacher-v2/log-%j
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

# Experiment loop
n_trains=(5 10 20 50 100 1000 10000 100000 1000000 10000000)
student_seeds=(1001 1002 1003 1004 1005 1006 1007 1008 1009)
for n_train in "\${n_trains[@]}"; do
  for student_seed in "\${student_seeds[@]}"; do
    # Run 8 experiments in parallel
    waitUntilNJobsRemain 8

    # Run experiment
    { LD_LIBRARY_PATH=/home/gridsan/groups/ccg/envs/scaling-v2/lib python -m src.student_teacher_v2.train --n_train \$n_train --student_seed \$student_seed $@; } &
  done
done

# Wait until all experiments finish.
waitUntilNJobsRemain 1

# Then kill mallory
pkill -P \$\$
EOT

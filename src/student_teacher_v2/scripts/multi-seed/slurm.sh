#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./src/student_teacher_v2/scripts/multi-seed/slurm.sh
#
# For details on how this script works, see the SLURM guide here:
# https://supercloud.mit.edu/submitting-jobs.
#
# Also see https://stackoverflow.com/a/44168719/1337463.

start_seed=$1
end_seed=$2
shift; shift;

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
for student_seed in \$(eval echo {$start_seed..$end_seed}); do
    # Run 12 experiments in parallel
    waitUntilNJobsRemain 12

    # Run experiment
    { LD_LIBRARY_PATH=/home/gridsan/groups/ccg/envs/scaling-v2/lib python -m src.student_teacher_v2.train --student_seed \$student_seed $@; } &
done

# Wait until all experiments finish.
waitUntilNJobsRemain 1

# Then kill mallory
pkill -P \$\$
EOT

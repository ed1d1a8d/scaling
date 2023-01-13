# Should be launched with cwd = root of repo.
import os
import subprocess
from typing import Sequence

import git.repo

GIT_ROOT = git.repo.Repo(".", search_parent_directories=True).working_tree_dir

SBATCH_TEMPLATE = """#!/bin/bash

# Slurm sbatch options
{GPU_STR}
#SBATCH -c {N_CPUS}
#SBATCH -o {LOG_DIR}/log-%j
#SBATCH -e {LOG_DIR}/log-%j

# Print info about the node we're running on
echo "Requested GPUS: {N_GPUS}"
echo "Requested CPUS: {N_CPUS}"
echo "GPU stats:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv
echo "Disk/memory utilization:"
df -h
echo "System info: $(uname -a)"
echo "Running as user: $(whoami)"
echo "Running in directory: $(pwd)"
echo "Start time: $(date)"
echo

# Load cuda 11.6
# See https://supercloud.mit.edu/submitting-jobs
module load cuda/11.6

# Setup conda-pack environment.
source scripts/setup-env.sh -r scaling-v3
source $DST_ENV_PATH/bin/activate

# Set up and launch proxy
source scripts/rand-mallory.sh
{{ mallory -config $TMP_MALLORY_CONFIG >/dev/null; }} &
sleep 10  # Wait for mallory to start

# Set up Hugging Face cache that supports locking.
export HF_HOME="/run/user/$UID/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE

# Utility function that waits until n background jobs are left running
# First argument is n
function waitUntilNJobsRemain() {{
  local n_jobs=$(jobs -rp | wc -l)
  echo "Waiting on $n_jobs background jobs until $1 jobs left..."
  while [[ $n_jobs -gt $1 ]]; do
    n_jobs=$(jobs -rp | wc -l)
    # echo -n "."  # Uncomment to print a dot for each sleep
    sleep 1
  done
  echo ""
}}

# Wait up to 5 minutes for gpu memory to clear (< 100 MiB)
# (needed due to wacky slurm behavior)
echo -n "Waiting for gpu memory to clear..."
for i in {{1..300}}; do
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "Current memory used: $mem_used MiB"
    if [[ $mem_used -lt 100 ]]; then
        echo "Memory cleared!"
        break
    fi
    sleep 1
done
if [[ $mem_used -ge 100 ]]; then
    echo "Memory not cleared!!!!"
fi

# Run experiment
{EXPERIMENT_CMD}

# Wait until only mallory is left
waitUntilNJobsRemain 1
pkill -P $$

# Delete conda-pack environment
rm -rf $DST_ENV_PATH

# Print ending time
echo
echo "End time: $(date)"
"""


def launch_single_experiment(
    command: str,
    log_dir: str,
    n_gpus: int,
    n_cpus: int,
) -> None:
    full_log_dir = os.path.abspath(
        os.path.expanduser(f"~/slurm-logs/scaling/{log_dir}")
    )

    # Ensure log_dir exists
    os.makedirs(full_log_dir, exist_ok=True)

    # Construct sbatch input
    sbatch_input = SBATCH_TEMPLATE.format(
        EXPERIMENT_CMD=command,
        LOG_DIR=full_log_dir,
        GPU_STR=f"#SBATCH --gres=gpu:volta:{n_gpus}"
        if n_gpus > 0
        else "# No gpus requested",
        N_GPUS=n_gpus,
        N_CPUS=n_cpus,
    )

    # Run sbatch with sbatch_input as input, with cwd = root of repo
    subprocess.run(
        ["sbatch"],
        input=sbatch_input,
        encoding="utf-8",
        shell=True,
        check=True,
        cwd=GIT_ROOT,
    )


def launch_parallel_experiments(
    commands: Sequence[str],
    max_concurrent: int,
    log_dir: str,
    n_gpus: int,
    n_cpus: int,
) -> None:
    command = f"\nwaitUntilNJobsRemain {max_concurrent}\n".join(
        "{ " + c + "; } &" for c in commands
    )

    launch_single_experiment(
        command=command,
        log_dir=log_dir,
        n_gpus=n_gpus,
        n_cpus=n_cpus,
    )


def launch_sharded_experiments(
    commands: Sequence[str],
    max_concurrent: int,
    n_nodes: int,
    log_dir: str,
    n_gpus: int,
    n_cpus: int,
) -> None:
    """
    Experiments run on n_nodes nodes.
    On each node, up to max_concurrent experiments run at the same time.
    """

    # Split experiment_cmds into n_nodes chunks
    chunks = [commands[i::n_nodes] for i in range(n_nodes)]

    # Launch each chunk on a separate node
    for chunk in chunks:
        launch_parallel_experiments(
            commands=chunk,
            max_concurrent=max_concurrent,
            log_dir=log_dir,
            n_gpus=n_gpus,
            n_cpus=n_cpus,
        )


if __name__ == "__main__":
    # Example usage
    launch_sharded_experiments(
        commands=[f"echo {x}" for x in range(20)],
        max_concurrent=3,
        n_nodes=2,
        log_dir="test",
        n_gpus=0,
        n_cpus=2,
    )

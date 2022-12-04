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

# Print info about the node we're running on
echo "Requested GPUS: {N_GPUS}"
echo "Requested CPUS: {N_CPUS}"
echo "Actual GPUS:"
nvidia-smi -L
echo
echo "System info: $(uname -a)"
echo "Running as user: $(whoami)"
echo "Running in directory: $(pwd)"
echo "Start time: $(date)"
echo

# Load conda environment.
# See https://github.com/conda/conda/issues/7980#issuecomment-441358406
# for details on why we do it this way.
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2021b/etc/profile.d/conda.sh
conda activate scaling-v2

# Set up environment variables.
export LD_LIBRARY_PATH=/home/gridsan/groups/ccg/envs/scaling-v2/lib

# Set up and launch proxy
source scripts/rand-mallory.sh
{{ mallory -config $TMP_MALLORY_CONFIG; }} &
sleep 10  # Wait for mallory to start

# Set up Hugging Face cache that supports locking.
export HF_HOME="/state/partition1/user/$USER/huggingface"
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
    echo -n "."
    sleep 1
  done
  echo ""
}}

# Run experiment
{EXPERIMENT_CMD}

# Wait until only mallory is left
waitUntilNJobsRemain 1
pkill -P $$

# Print ending time
echo
echo "End time: $(date)"
"""


def launch_single_experiment(
    command: str,
    log_dir: str,
    n_gpus: int = 1,
    n_cpus: int = 20,
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
        GPU_STR=f"#SBATCH --gres=gpu:{n_gpus}"
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
    n_gpus: int = 1,
    n_cpus: int = 20,
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
    n_gpus: int = 1,
    n_cups: int = 20,
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
            n_cpus=n_cups,
        )

#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/migrations/eval_with_autoattack/go.sh
start_idxs=(0 20 40 80 100 120 140 160 180 200)
for start_idx in "${start_idxs[@]}"; do
    ./src/ax/migrations/eval_with_autoattack/slurm.sh \
        --row_start $start_idx \
        --row_end $(expr $start_idx + 20)
done

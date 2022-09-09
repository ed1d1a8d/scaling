#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/student_teacher_v2/scripts/multi-seed/go.sh

start_seeds=(1000 2000 3000 4000 5000 6000 7000 8000 9000)
for start_seed in "${start_seeds[@]}"; do
  end_seed=$(expr $start_seed + 999)
  ./src/student_teacher_v2/scripts/multi-seed/slurm.sh \
    $start_seed $end_seed \
    --tags local-minima-8-96-192-1 \
    --teacher_seed 101 \
    --n_train -1 \
    --n_val 10000 \
    --n_test 1000000 \
    --samples_per_eval 100000 \
    --wandb_dir
done

#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/student_teacher_v2/scripts/go.sh

width_scales=(1 5 10 20 100 1000)
for width_scale in "${width_scales[@]}"; do
  ./src/student_teacher_v2/scripts/slurm.sh \
    --tags overparam-scaling-4-2-1-try2 \
    --optimizer SGD \
    --lr 1e-2 \
    --n_workers 0 \
    --n_val 10000 \
    --n_test 1000000 \
    --samples_per_eval 100000 \
    --student_width_scale_factor $width_scale \
    --student_seed $student_seed
done

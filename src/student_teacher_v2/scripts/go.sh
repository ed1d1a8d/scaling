#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/student_teacher_v2/scripts/go.sh

width_scales=(1 10 100 1000)
student_seeds=(9000 9001 9002)
for width_scale in "${width_scales[@]}"; do
  for student_seed in "${student_seeds[@]}"; do
  ./src/student_teacher_v2/scripts/slurm.sh \
    --tags overparam-scaling-4-2-1 \
    --n_val 10000 \
    --n_test 1000000 \
    --samples_per_eval 100000 \
    --student_width_scale_factor $width_scale \
    --student_seed $student_seed
  done
done

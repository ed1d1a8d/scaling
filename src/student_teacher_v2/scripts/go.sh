#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/student_teacher_v2/scripts/go.sh

width_scales=(0.1 0.5 1 10 100)
for width_scale in "${width_scales[@]}"; do
  ./src/student_teacher_v2/scripts/slurm.sh \
    --tags double-scaling-8-96-192-1 \
    --input_dim 8 \
    --teacher_widths 96 192 1 \
    --teacher_seed 101 \
    --n_val 10000 \
    --n_test 1000000 \
    --samples_per_eval 1000000 \
    --student_width_scale_factor $width_scale
done

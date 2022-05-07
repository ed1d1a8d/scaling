#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/student_teacher/scripts/go-sparsity.sh

n_trains=(32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 128000)
width_scales=(1)
l1_reg_lambdas=(0 3e-7)
for n_train in "${n_trains[@]}"
do
  for width_scale in "${width_scales[@]}"
  do
    for l1_reg_lambda in "${l1_reg_lambdas[@]}"
    do
        ./src/experiments/student_teacher/scripts/slurm.sh \
            --n_train $n_train \
            --student_width_scale $width_scale \
            --l1_reg_lambda $l1_reg_lambda \
            --tags sparsity_v1
    done
  done
done

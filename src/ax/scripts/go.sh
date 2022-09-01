#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh

# 200 300 500 1000 2000 5000 10000 20000 50000 200000 500000 2000000 5942688
n_trains=(1000 2000 5000 10000 20000 50000 200000 500000 2000000 5942688)
adv_trains=(False True)
use_softmaxes=(False)

for n_train in "${n_trains[@]}"; do
  for adv_train in "${adv_trains[@]}"; do
    for use_softmax in "${use_softmaxes[@]}"; do
      ./src/ax/scripts/slurm.sh \
        --dataset CIFAR5m \
        --use_teacher True \
        --n_train $n_train \
        --do_adv_training $adv_train \
        --teacher_use_softmax $use_softmax \
        --tags student-teacher-random-init
    done
  done
done

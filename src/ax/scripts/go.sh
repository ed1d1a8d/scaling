#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh
n_trains=(50 80 100 200 300 500 1000 2000 5000 10000 20000 50000)
widths=(10 20)
for width in "${widths[@]}"; do
  for n_train in "${n_trains[@]}"; do
    ./src/ax/scripts/slurm.sh \
      --dataset CIFAR10 \
      --width $width \
      --n_train $n_train \
      --do_adv_training False \
      --tags try1.3

    ./src/ax/scripts/slurm.sh \
      --dataset CIFAR10 \
      --width $width \
      --n_train $n_train \
      --do_adv_training True \
      --tags try1.3
  done
done

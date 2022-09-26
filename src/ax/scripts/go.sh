#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh

launch_series () {
  n_trains=(500 1000 2000 5000 10000 20000 50000 200000 500000 2000000 5942688)
  for n_train in "${n_trains[@]}"; do
    ./src/ax/scripts/slurm.sh \
      --tags smaller-wrns \
      --dataset CIFAR5m \
      --do_adv_training True \
      --n_train $n_train \
      --depth $1 \
      --width $2
  done
}

launch_series 28 5
launch_series 16 10

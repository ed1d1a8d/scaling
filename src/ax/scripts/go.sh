#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh
n_trains=(2000 5000 10000 20000 50000 200000)
adv_eps_trains=(4 6 10 12 14 16)
for n_train in "${n_trains[@]}"; do
  for adv_eps_train in "${adv_eps_trains[@]}"; do
    ./src/ax/scripts/slurm.sh \
      --dataset CIFAR5m \
      --width 10 \
      --do_adv_training True \
      --n_train $n_train \
      --adv_eps_train $(echo $adv_eps_train / 255 | bc -l) \
      --tags eps-ablation-v1
  done
done

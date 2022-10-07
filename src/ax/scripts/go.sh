#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh

# 200 300 500 1000 2000 5000 10000 20000 50000 200000 500000 2000000 5942688
n_trains=(1000 3000 10000 30000 100000 300000 1000000 3000000 10000000 20000000)
adv_trains=(True)

for n_train in "${n_trains[@]}"; do
  for adv_train in "${adv_trains[@]}"; do
    ./src/ax/scripts/slurm.sh \
      --dataset MNIST20m \
      --depth 16 \
      --width 5 \
      --adv_eps_train 0.3 \
      --adv_eps_eval 0.3 \
      --pgd_steps 10 \
      --n_train $n_train \
      --do_adv_training $adv_train \
      --tags mnist20m
  done

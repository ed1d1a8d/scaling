#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/param-scaling.sh

depths=(28 34 40 46 52)
widths=(10 20)

for depth in "${depths[@]}"; do
  for width in "${widths[@]}"; do
    ./src/ax/scripts/slurm.sh \
      --tags param-scaling-v1 \
      --dataset CIFAR5m \
      --n_train 5942688 \
      --do_adv_training True \
      --batch_size 256 \
      --eval_batch_size 256 \
      --depth $depth \
      --width $width
  done
done

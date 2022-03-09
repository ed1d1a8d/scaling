#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/quantized_harmonics/scripts/go.sh

seeds = (0 1 2 3 4 5)
for seed in "${seeds[@]}"
do
  ./src/experiments/quantized_harmonics/scripts/slurm.sh --train_seed $seed
done

#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

n_trains=(32 50 80 100 128 200 400 800 1600 3200 6400 10000 20000 40000 80000)
regs=(MCLS NONE)
for n_train in "${n_trains[@]}"
do
  for reg in "${regs[@]}"
  do
    ./src/experiments/harmonics/scripts/slurm.sh \
      --n_train $n_train \
      --high_freq_reg $reg \
      --tags nn
  done
done

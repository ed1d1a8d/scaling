#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh

n_trains=(50000 500000 5942688)
for n_train in "${n_trains[@]}"; do
  ./src/ax/scripts/slurm.sh \
    --n_trains $n_train \
    --do_adv_training False \
    --tags try1

  ./src/ax/scripts/slurm.sh \
    --n_trains $n_train \
    --do_adv_training True \
    --tags try1
done

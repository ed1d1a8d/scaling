#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh

n_trains=(10000 50000 500000 5942688)
for n_train in "${n_trains[@]}"; do
  ./src/ax/scripts/slurm.sh \
    --n_train $n_train \
    --do_adv_training False \
    --width 20 \
    --tags try1.2.1

  ./src/ax/scripts/slurm.sh \
    --n_train $n_train \
    --do_adv_training True \
    --width 20 \
    --tags try1.2.1
done

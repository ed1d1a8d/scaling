#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/pretrain/scripts/go.sh

n_trains=(10 20 50 100 200 500 1000 2000 5000 10000 20000 45000)

for n_train in "${n_trains[@]}"; do
  ./src/ax/scripts/go.sh \
    --n_train $n_train \
    --tags clip-cifar-finetune
done

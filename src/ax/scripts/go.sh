#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/go.sh
datasets=(SquareCircle HVStripe LightDark)
widths=(10 20)
adv_trains=(False True)
for dataset in "${datasets[@]}"; do
  for width in "${widths[@]}"; do
    for adv_train in "${adv_trains[@]}"; do
      ./src/ax/scripts/slurm.sh \
        --samples_per_eval 10240 \
        --dataset $dataset \
        --width $width \
        --do_adv_training $adv_train \
        --tags try1.3
    done
  done
done

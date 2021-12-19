#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

freq_limits=( 2 3 )
ncs=( 4 16 )
for freq_limit in "${freq_limits[@]}"
do
  for nc in "${ncs[@]}"
  do
      ./src/experiments/harmonics/scripts/slurm.sh \
                --freq_limit $freq_limit \
                --num_components $nc
  done
done

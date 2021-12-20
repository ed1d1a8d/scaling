#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

high_freq_lambdas=( 0.01 0.1 1 10 100 )
for high_freq_lambda in "${high_freq_lambdas[@]}"
do
  ./src/experiments/harmonics/scripts/slurm.sh \
            --high_freq_lambda $high_freq_lambda
done

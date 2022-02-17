#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

high_freq_lambdas=(0 1)
high_freq_mcls_sampless=(1024 4096 10000 20000)
high_freq_dft_sss=(32 64 128)
for high_freq_lambda in "${high_freq_lambdas[@]}"
do
  for samples in "${high_freq_mcls_sampless[@]}"
  do
    ./src/experiments/harmonics/scripts/slurm.sh \
              --high_freq_reg MCLS \
              --high_freq_lambda $high_freq_lambda \
              --high_freq_mcls $samples
  done

  for ss in "${high_freq_dft_sss[@]}"
  do
    ./src/experiments/harmonics/scripts/slurm.sh \
              --high_freq_reg DFT \
              --high_freq_lambda $high_freq_lambda \
              --high_freq_dft_ss $ss
  done
done

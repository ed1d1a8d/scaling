#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

input_dims=( 2 4 8 )
freq_limits=( 4 8 16 )
layer_width_lists=( "96 192 1" "192 384 1" "384 768 1")
for input_dim in "${input_dims[@]}"
do
	for freq_limit in "${freq_limits[@]}"
  do
		for layer_widths in "${layer_width_lists[@]}"
	  do
			./src/experiments/harmonics/slurm.sh \
                --input_dim $input_dim \
                --freq_limit $freq_limit \
                --layer_widths $layer_widths
	  done
  done
done

#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

input_dims=( 2 4 8 )
freq_limits=( 4 8 )
layer_width_lists=( "96 192 1" "384 768 1")
learning_rates=( "0.003" "0.0003" )
for input_dim in "${input_dims[@]}"
do
	for freq_limit in "${freq_limits[@]}"
  do
		for layer_widths in "${layer_width_lists[@]}"
	  do
      for learning_rate in "${learning_rates[@]}"
	    do
        ./src/experiments/harmonics/scripts/slurm.sh \
                  --input_dim $input_dim \
                  --freq_limit $freq_limit \
                  --layer_widths $layer_widths \
                  --learning_rate $learning_rate
      done
	  done
  done
done

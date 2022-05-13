#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

widths=(128 256)
freq_limits=(2 4 6)
for width in "${widths[@]}"; do
  for hffl in "${freq_limits[@]}"; do
    ./src/experiments/harmonics/scripts/slurm.sh \
      --layer_widths $width $width $width 1 \
      --high_freq_reg MCLS \
      --high_freq_freq_limit $hffl \
      --tags nn
  done

  ./src/experiments/harmonics/scripts/slurm.sh \
    --layer_widths $width $width $width 1 \
    --high_freq_reg NONE \
    --tags nn
done

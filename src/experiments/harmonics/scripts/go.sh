#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/experiments/harmonics/scripts/go.sh

widths=(128 256 512)
freq_limits=(2 3 4)
for width in "${widths[@]}"; do
  for hffl in "${freq_limits[@]}"; do
    ./src/experiments/harmonics/scripts/slurm.sh \
      --net_width $width \
      --high_freq_reg MCLS \
      --high_freq_freq_limit $hffl \
      --tags nn cliff_v2
  done

  ./src/experiments/harmonics/scripts/slurm.sh \
    --net_width $width \
    --high_freq_reg NONE \
    --high_freq_freq_limit 0 \
    --tags nn cliff_v2
done

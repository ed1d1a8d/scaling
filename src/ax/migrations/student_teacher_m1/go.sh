#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./src/ax/migrations/student_teacher_m1/go.sh
row_mods=(0 1 2 3 4 5 6 7)
for row_mod in "${row_mods[@]}"; do
    ./src/ax/migrations/student_teacher_m1/slurm.sh \
        --row_mod $row_mod \
        --num_jobs 8
done

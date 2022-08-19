#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./src/ax/scripts/cifar-hypers.sh

function launch_run() {
  ./src/ax/scripts/slurm.sh \
    --dataset CIFAR10 \
    --n_train 50000 \
    --do_adv_training False \
    --pgd_steps 0 \
    --tags cifar10-hyper-v0 \
    $@
}

launch_run
launch_run --batch_size 128
launch_run --data_augmentation True
launch_run --optimizer SGD --lr 0.1
launch_run --batch_size 128 --data_augmentation True --optimizer SGD --lr 0.1

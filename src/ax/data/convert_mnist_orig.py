# How to use this script:
# 1. Download infimnist from https://leon.bottou.org/projects/infimnist
# 2. Generate the infimnist files as follows (you can substitue any number for 8109999 below, depending on how much training data you want to generate):
# $ infimnist lab 0 9999 > test10k-labels
# $ infimnist pat 0 9999 > test10k-patterns
# $ infimnist lab 10000 8109999 > mnist8m-labels-idx1-ubyte
# $ infimnist pat 10000 8109999 > mnist8m-patterns-idx3-ubyte
# 3. Copy the files into scaling/data/mnist8m
# 4. Run this script (make sure the filenames variable below matches the filenames you generated in step 2)

import os

import git.repo
import numpy as np
from ffcv.fields import IntField, NDArrayField
from ffcv.writer import DatasetWriter
from mlxtend.data import loadlocal_mnist

GIT_ROOT = str(
    git.repo.Repo(".", search_parent_directories=True).working_tree_dir
)


class NumpyDataset:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        assert xs.shape[0] == ys.shape[0]

        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx: int):
        return (self.xs[idx], self.ys[idx])


def get_mnist_datasets(
    xs_test_file: str,
    ys_test_file: str,
    xs_train_file: str,
    ys_train_file: str,
    n_val: int = 5_000,
):
    xs_test, ys_test = loadlocal_mnist(
        images_path=xs_test_file, labels_path=ys_test_file
    )
    print("Loaded test files")

    xs, ys = loadlocal_mnist(
        images_path=xs_train_file, labels_path=ys_train_file
    )
    print("Loaded train files")

    xs_train, ys_train = (xs[:-n_val], ys[:-n_val])

    xs_val, ys_val = (xs[-n_val:], ys[-n_val:])

    return {
        "train": NumpyDataset(xs_train, ys_train),
        "val": NumpyDataset(xs_val, ys_val),
        "test": NumpyDataset(xs_test, ys_test),
    }


filenames = {
    "xs_test": "test10k-patterns",
    "ys_test": "test10k-labels",
    "xs_train": "mnist60k-patterns-idx3-ubyte",
    "ys_train": "mnist60k-labels-idx1-ubyte",
}

rel_path = "data/mnist-orig/"
paths = {}
for (name, filename) in filenames.items():
    path = os.path.join(GIT_ROOT, rel_path + filename)
    paths[name] = path


datasets = get_mnist_datasets(
    paths["xs_test"], paths["ys_test"], paths["xs_train"], paths["ys_train"]
)

for (name, ds) in datasets.items():
    writer = DatasetWriter(
        os.path.join(GIT_ROOT, rel_path + name + ".beton"),
        {
            "image": NDArrayField(np.dtype(np.uint8), (784,)),
            "label": IntField(),
        },
        num_workers=100,
    )
    writer.from_indexed_dataset(ds, chunksize=10000)

# * Generating files containing the standard MNIST testing set:

#   $ infimnist lab 0 9999 > test10k-labels
#   $ infimnist pat 0 9999 > test10k-patterns

# * Generating files containing the standard MNIST training set:

#   $ infimnist lab 10000 69999 > mnist60k-labels
#   $ infimnist pat 10000 69999 > mnist60k-patterns

# * Generating files containing the MNIST8M training set:

#   $ infimnist lab 10000 8109999 > mnist8m-labels
#   $ infimnist pat 10000 8109999 > mnist8m-patterns

import os

import git
import numpy as np
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from numpy.lib.npyio import NpzFile

GIT_ROOT = str(git.Repo(".", search_parent_directories=True).working_tree_dir)


class NumpyDataset:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        assert xs.shape[0] == ys.shape[0]

        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx: int):
        return (self.xs[idx], self.ys[idx])


def get_cifar5m_part(i: int) -> str:
    return os.path.join(GIT_ROOT, f"data/cifar5m/part{i}.npz")


def get_cifar5m_datasets(files: list[str], n_test: int = 30000):
    npzs: list[NpzFile] = [np.load(f) for f in files]
    print("npz files all loaded!")

    xs = np.concatenate([npz["X"] for npz in npzs])
    ys = np.concatenate([npz["Y"] for npz in npzs])
    del npzs  # Free memory
    print("concatenation finished")

    xs_train, ys_train = xs[:-n_test], ys[:-n_test]
    xs_test, ys_test = xs[-n_test:], ys[-n_test:]

    print(len(xs_test), len(ys_test))
    print(np.unique(ys_test, return_counts=True))

    return {
        "train": NumpyDataset(xs_train, ys_train),
        "test": NumpyDataset(xs_test, ys_test),
    }


datasets = get_cifar5m_datasets([get_cifar5m_part(i) for i in range(6)])

for (name, ds) in datasets.items():
    writer = DatasetWriter(
        os.path.join(GIT_ROOT, f"data/cifar5m/{name}.beton"),
        {
            "image": RGBImageField(write_mode="raw"),
            "label": IntField(),
        },
        num_workers=100,
    )
    writer.from_indexed_dataset(ds, chunksize=10000)

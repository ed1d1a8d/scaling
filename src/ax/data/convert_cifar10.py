import os

import git.repo
import torchvision
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter

GIT_ROOT = str(
    git.repo.Repo(".", search_parent_directories=True).working_tree_dir
)

datasets = {
    "train": torchvision.datasets.CIFAR10(
        "/var/tmp", train=True, download=True
    ),
    "test": torchvision.datasets.CIFAR10(
        "/var/tmp", train=False, download=True
    ),
}

for name, ds in datasets.items():
    writer = DatasetWriter(
        os.path.join(GIT_ROOT, f"data/cifar5m/{name}-orig.beton"),
        {"image": RGBImageField(write_mode="raw"), "label": IntField()},
        num_workers=20,
    )
    writer.from_indexed_dataset(ds)

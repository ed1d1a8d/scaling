from typing import Type

from src.pretrain.datasets.base import BaseDatasetConfig
from src.pretrain.datasets.vision import cifar10, cifar100, imagenette, svhn


def get_dataset_index() -> dict[str, Type[BaseDatasetConfig]]:
    return {
        "cifar10": cifar10.CIFAR10,
        "cifar100": cifar100.CIFAR100,
        "imagenette": imagenette.Imagenette,
        "svhn": svhn.SVHN,
    }


def get_dataset_key(dataset_cfg: BaseDatasetConfig) -> str:
    for key, cfg_t in get_dataset_index().items():
        if isinstance(dataset_cfg, cfg_t):
            return key
    raise ValueError(f"Dataset config not found: {dataset_cfg}")

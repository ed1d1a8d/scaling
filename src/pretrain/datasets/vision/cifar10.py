import dataclasses
from typing import Callable, Optional

import torch.utils.data
import torchvision

from src.pretrain.datasets.base import BaseDatasetConfig


@dataclasses.dataclass
class CIFAR10(BaseDatasetConfig):
    id: str = "cifar10"

    @property
    def class_names(self) -> tuple[str, ...]:
        return (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        return torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )

    def get_test_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        return torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform,
        )

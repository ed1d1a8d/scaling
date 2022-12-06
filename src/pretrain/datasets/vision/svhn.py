import dataclasses
from typing import Callable, Optional

import torch.utils.data
import torchvision

from src.pretrain.datasets import BaseDatasetConfig


@dataclasses.dataclass
class SVHN(BaseDatasetConfig):
    id: str = "svhn"

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(10))

    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        # TODO: Possible include the 'extra' training dataset?
        return torchvision.datasets.SVHN(
            root=self.data_dir,
            split="train",
            download=True,
            transform=transform,
        )

    def get_test_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        return torchvision.datasets.SVHN(
            root=self.data_dir,
            split="test",
            download=True,
            transform=transform,
        )

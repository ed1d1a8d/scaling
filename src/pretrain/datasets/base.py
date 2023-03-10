import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
import torch.utils.data


@dataclasses.dataclass
class BaseDatasetConfig(ABC):
    """
    No validation dataset is included. This is because the validation dataset
    counts as training data from an information-theoretic perspective.

    If you need a validation dataset, please subset the training dataset.
    A good strategy for this is to take ~5% of the training dataset as the
    validation dataset.
    """

    id: str
    data_dir: str = "/projects/scaling/data-downloads"

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def parse_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple of (xs, ys)."""
        return batch

    @abstractmethod
    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_test_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

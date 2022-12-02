"""
WARNING: We load Imagenette via huggingface datasets. In order to use
         huggingface on supercloud, you need to soft link ~/.cache/huggingface
         to a directory that supports locking.

         It is recommended that you symlink to
         /state/partition1/user/$USER/huggingface.

         If launching from a slurm script, you should make sure that the
         symlinked directory exists before running the code in this file.
         This can be done by running the command:

            mkdir -p /state/partition1/user/$USER/huggingface
"""

import dataclasses
from typing import Callable, Optional

import torch.utils.data
from datasets.load import load_dataset

from src.pretrain.datasets import BaseDatasetConfig


@dataclasses.dataclass
class Imagenette(BaseDatasetConfig):
    id: str = "imagenette"

    def class_names(self) -> tuple[str, ...]:
        return (
            "tench",
            "English springer",
            "cassette player",
            "chain saw",
            "church",
            "French horn",
            "garbage truck",
            "gas pump",
            "golf ball",
            "parachute",
        )

    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        return load_dataset(
            "frgfm/imagenette",
            "320px",
            split="train",
        ).set_transform(  # type: ignore
            transform
        )

    def get_test_ds(
        self, transform: Optional[Callable] = None,
    ) -> torch.utils.data.Dataset:
        return load_dataset(
            "frgfm/imagenette",
            "320px",
            split="validation",
        ).set_transform(  # type: ignore
            transform
        )

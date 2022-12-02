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
from typing import Any, Callable, Optional

import torch
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

    def parse_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["image"], batch["label"]

    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        ds = load_dataset(
            "frgfm/imagenette",
            "320px",
            split="train",
        )

        ds.set_transform(  # type: ignore
            lambda d: d
            | (
                {}
                if transform is None
                else {
                    "image": transform(d["image"][0]).unsqueeze(
                        0
                    ),  # We unsqueeze to account for pytorch collation.
                }
            )
        )

        return ds  # type: ignore

    def get_test_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        ds = load_dataset(
            "frgfm/imagenette",
            "320px",
            split="validation",
        )

        ds.set_transform(  # type: ignore
            lambda d: d
            | (
                {}
                if transform is None
                else {
                    "image": transform(d["image"][0]).unsqueeze(
                        0
                    ),  # We unsqueeze to account for pytorch collation.
                }
            )
        )

        return ds  # type: ignore

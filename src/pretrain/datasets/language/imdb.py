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
class Imdb(BaseDatasetConfig):
    id: str = "imdb"

    @property
    def class_names(self) -> tuple[str, ...]:
        return (
            "neg",
            "pos",
        )

    def parse_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["text"], batch["label"]

    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        ds = load_dataset("imdb", split="train")

        return ds  # type: ignore

    def get_test_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        ds = load_dataset(
            "imdb",
            split="test",
        )

        return ds  # type: ignore

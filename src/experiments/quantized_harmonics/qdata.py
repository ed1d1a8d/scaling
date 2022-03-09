from __future__ import annotations

from typing import Callable

import pytorch_lightning as pl
import torch
import torch.random
from torch.utils.data import DataLoader, TensorDataset
import einops
import numpy as np


class QDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        input_dim: int,
        input_resolution: int,
        n_train: int,
        seed: int = -1,
        batch_size: int = 256,
        gpus: int = 1,
        num_workers: int = 16,
    ):
        super().__init__()
        self.fn = fn
        self.input_dim = input_dim
        self.input_resolution = input_resolution

        self.n_train = n_train
        assert 0 <= n_train <= self.n_datapoints

        self.seed = seed

        self.batch_size = batch_size
        self.gpus = gpus
        self.num_workers = num_workers

    @property
    def n_datapoints(self):
        return self.input_resolution ** self.input_dim

    def get_ds(self, xs: torch.Tensor) -> TensorDataset:
        xs_coords = (xs + 0.5)/self.input_resolution
        ys = torch.concat(
            [
                self.fn(batch_xs)
                for (batch_xs,) in DataLoader(
                    TensorDataset(xs_coords),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
            ]
        )
        return TensorDataset(xs, ys)

    def setup(self, *_, **__):
        all_xs = torch.tensor(
            einops.rearrange(
                np.mgrid[
                    tuple(
                        slice(0, self.input_resolution)
                        for _ in range(self.input_dim)
                    )
                ],
                "input_dim ... -> (...) input_dim",
            )
        )
        all_xs_shuffled = all_xs[
            torch.randperm(
                all_xs.shape[0],
                generator=torch.Generator().manual_seed(self.seed),
            )
        ]

        self.train_ds = self.get_ds(all_xs_shuffled[: self.n_train])
        self.val_ds = self.get_ds(all_xs_shuffled[self.n_train :])

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

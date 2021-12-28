from __future__ import annotations

from typing import Callable

import pytorch_lightning as pl
import torch
import torch.random
from torch.utils.data import DataLoader, TensorDataset


class HypercubeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        input_dim: int,
        n_train: int,
        n_val: int,
        train_seed: int = -1,
        val_seed: int = -2,
        batch_size: int = 256,
        gpus: int = 1,
        num_workers: int = 16,
    ):
        super().__init__()
        self.fn = fn
        self.input_dim = input_dim

        self.n_train = n_train
        self.n_val = n_val

        self.train_seed = train_seed
        self.val_seed = val_seed

        self.batch_size = batch_size
        self.gpus = gpus
        self.num_workers = num_workers

    def get_ds(self, n_samples: int, seed: int) -> TensorDataset:
        xs = torch.rand(
            (n_samples, self.input_dim),
            generator=torch.Generator().manual_seed(seed),
        )
        ys = torch.concat(
            [
                self.fn(batch[0])
                for batch in DataLoader(
                    TensorDataset(xs),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
            ]
        )
        return TensorDataset(xs, ys)

    def setup(self, *_, **__):
        self.train_ds = self.get_ds(self.n_train, self.train_seed)
        self.val_ds = self.get_ds(self.n_val, self.val_seed)

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.train_seed),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

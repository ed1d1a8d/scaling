from __future__ import annotations

from typing import Callable
import dataclasses

import pytorch_lightning as pl
import torch
import torch.random
from torch.utils.data import DataLoader, TensorDataset


@dataclasses.dataclass
class HypercubeDataModuleConfig:
    input_dim: int
    n_train: int
    n_val: int
    train_seed: int = -1
    val_seed: int = -2
    batch_size: int = 256
    gpus: int = 1
    num_workers: int = 16

    cube_lo: float = 0.0
    cube_hi: float = 1.0

    def __post_init__(self):
        assert self.cube_lo <= self.cube_hi


class HypercubeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        cfg: HypercubeDataModuleConfig,
    ):
        super().__init__()
        self.fn = fn
        self.cfg = cfg

    def get_ds(self, n_samples: int, seed: int) -> TensorDataset:
        xs = (
            torch.rand(
                (n_samples, self.cfg.input_dim),
                generator=torch.Generator().manual_seed(seed),
            )
            * (self.cfg.cube_hi - self.cfg.cube_lo)
            + self.cfg.cube_lo
        )
        with torch.no_grad():
            ys = torch.concat(
                [
                    self.fn(batch[0])
                    for batch in DataLoader(
                        TensorDataset(xs),
                        batch_size=self.cfg.batch_size,
                        shuffle=False,
                    )
                ]
            )
        return TensorDataset(xs, ys)

    def setup(self, *_, **__):
        self.train_ds = self.get_ds(self.cfg.n_train, self.cfg.train_seed)
        self.val_ds = self.get_ds(self.cfg.n_val, self.cfg.val_seed)

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.cfg.train_seed),
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

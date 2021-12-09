import dataclasses
from typing import Optional

import pytorch_lightning as pl
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random
from torch.utils.data import DataLoader, TensorDataset


@dataclasses.dataclass(frozen=True)
class HarmonicFnConfig:
    input_dim: int
    freq_limit: int
    num_components: int
    seed: int = 42


class HarmonicFn(pl.LightningModule):
    """
    Represents a bandlimited function with of a fixed number of
    integer-frequency harmonics.
    The function is scaled so as to be periodic on the unit-square.
    """

    def __init__(self, cfg: HarmonicFnConfig):
        """Creates a random harmonic function."""
        super().__init__()

        self.cfg = cfg

        rng = torch.Generator().manual_seed(cfg.seed)

        self.coeffs = nn.Parameter(
            F.normalize(
                dim=0,
                input=torch.normal(
                    size=(cfg.num_components,),
                    mean=0,
                    std=1,
                    generator=rng,
                ),
            ),
            requires_grad=False,
        )

        self.freqs = nn.Parameter(
            torch.randint(
                size=(cfg.num_components, cfg.input_dim),
                low=0,
                high=cfg.freq_limit + 1,
                generator=rng,
            ).float(),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.cfg.input_dim

        # Could also do exp
        return torch.cos(2 * torch.pi * x @ self.freqs.T) @ self.coeffs

    def viz_2d(self, side_samples: int):
        assert self.cfg.input_dim == 2
        utils.viz_2d(
            self.forward,
            side_samples=side_samples,
        )


class HarmonicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hf: HarmonicFn,
        n_train: int,
        n_val: int,
        train_seed: int = -1,
        val_seed: int = -2,
        batch_size: int = 256,
        gpus: int = 1,
        num_workers: int = 16,
    ):
        super().__init__()
        self.hf = hf

        self.n_train = n_train
        self.n_val = n_val

        self.train_seed = train_seed
        self.val_seed = val_seed

        self.batch_size = batch_size
        self.gpus = gpus
        self.num_workers = num_workers

    def get_ds(self, n_samples: int, seed: int) -> TensorDataset:
        xs = torch.rand(
            (n_samples, self.hf.cfg.input_dim),
            generator=torch.Generator().manual_seed(seed),
        )
        ys = torch.concat(
            [
                self.hf(batch[0])
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

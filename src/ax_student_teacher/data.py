import dataclasses
from math import ceil
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.utils.data
import tensorcanvas

T = TypeVar("T")


class SyntheticDS(torch.utils.data.Dataset, Generic[T]):
    def __init__(self, cfg: T, split: str, size: int):
        self.cfg = cfg
        self.split = split
        self.size = size

    def __len__(self):
        return self.size

    def get_rng(self, idx: int):
        return torch.Generator().manual_seed(hash(f"{self.split}+{idx}"))


@dataclasses.dataclass
class HyperCubeDSConfig:
    dim: int
    data_dim: int

    def __post_init__(self):
        assert self.data_dim <= self.dim


class HyperCubeDS(SyntheticDS[HyperCubeDSConfig]):
    def __getitem__(self, idx: int):
        rng = self.get_rng(idx)



        y: torch.Tensor = torch.tensor(idx % 2)

        x_base = torch.full(
            size=(
                self.cfg.n_channels,
                self.cfg.side_length,
                self.cfg.side_length,
            ),
            fill_value=(
                self.cfg.lo_intensity
                if y.item() == 0
                else self.cfg.hi_intensity
            ),
        )
        x = (
            x_base
            + self.cfg.noise_std * torch.randn(size=x_base.shape, generator=rng)
        ).clip(min=0, max=1)

        return x, y

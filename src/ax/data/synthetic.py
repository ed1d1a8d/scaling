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
class LightDarkDSConfig:
    side_length: int = 32
    n_channels: int = 3

    lo_intensity: float = 0.2
    hi_intensity: float = 0.8
    noise_std: float = 0.01


class LightDarkDS(SyntheticDS[LightDarkDSConfig]):
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


@dataclasses.dataclass
class HVStripeDSConfig:
    side_length: int = 32
    n_channels: int = 3

    background_intensity: float = 0.2
    stripe_intensity: float = 0.8
    stripe_frac: float = 0.1  # What fraction of the image the stripe takes up
    edge_pad_frac: float = 0.1  # Closest stripe can lie to an edge
    noise_std: float = 0.01

    def __post_init__(self):
        assert 0 <= self.edge_pad_frac < 1
        assert 0 < self.stripe_frac < 1


class HVStripeDS(SyntheticDS[HVStripeDSConfig]):
    """
    Dataset with horizontal and vertical stripes.
    Class 0 is horizontal.
    Class 1 is vertical.
    """

    @property
    def stripe_width(self) -> int:
        """In terms of pixels."""
        return int(self.cfg.stripe_frac * self.cfg.side_length)

    @property
    def edge_pad(self) -> int:
        """In terms of pixels."""
        return ceil(self.cfg.edge_pad_frac * self.cfg.side_length)

    def __getitem__(self, idx: int):
        rng = self.get_rng(idx)
        y: torch.Tensor = torch.tensor(idx % 2)

        stripe_location: int = torch.randint(
            low=self.edge_pad,
            high=self.cfg.side_length - self.edge_pad - self.stripe_width + 1,
            size=(),
            generator=rng,
        ).item()  # type: ignore

        x_base = torch.full(
            size=(
                self.cfg.n_channels,
                self.cfg.side_length,
                self.cfg.side_length,
            ),
            fill_value=self.cfg.background_intensity,
        )

        stripe_slice = slice(
            stripe_location, stripe_location + self.stripe_width
        )
        if y.item() == 0:
            x_base[:, :, stripe_slice] = self.cfg.stripe_intensity
        else:
            x_base[:, stripe_slice, :] = self.cfg.stripe_intensity

        x = (
            x_base
            + self.cfg.noise_std * torch.randn(size=x_base.shape, generator=rng)
        ).clip(min=0, max=1)

        return x, y


@dataclasses.dataclass
class SquareCircleDSConfig:
    side_length: int = 32
    n_channels: int = 3

    background_intensity: float = 0.2
    shape_intensity: float = 0.8
    edge_pad_frac: float = 0.1  # Closest shape can lie to an edge
    noise_std: float = 0.01

    square_len_frac: float = 0.2
    circle_rad_frac: float = 0.2 / np.sqrt(np.pi)

    def __post_init__(self):
        assert 0 < self.edge_pad_frac < 1
        assert 0 < self.square_len_frac < 1
        assert 0 < self.circle_rad_frac < 1


class SquareCircleDS(SyntheticDS[SquareCircleDSConfig]):
    """
    Dataset with squares and circles.
    Class 0 is square.
    Class 1 is circle.
    """

    @property
    def square_len(self) -> int:
        """In terms of pixels."""
        return int(self.cfg.square_len_frac * self.cfg.side_length)

    @property
    def circle_rad(self) -> int:
        """In terms of pixels."""
        return int(self.cfg.circle_rad_frac * self.cfg.side_length)

    @property
    def edge_pad(self) -> int:
        """In terms of pixels."""
        return ceil(self.cfg.edge_pad_frac * self.cfg.side_length)

    def __getitem__(self, idx: int):
        rng = self.get_rng(idx)
        y: torch.Tensor = torch.tensor(idx % 2)

        # Lower left corner
        shape_bbox_len: int = (
            self.square_len if y.item() == 0 else 2 * self.circle_rad
        )
        shape_coords: torch.Tensor = torch.randint(
            low=self.edge_pad,
            high=self.cfg.side_length - self.edge_pad - shape_bbox_len + 1,
            size=(2,),
            generator=rng,
        )
        shape_x: int = shape_coords[0].item()  # type: ignore
        shape_y: int = shape_coords[1].item()  # type: ignore

        x_base = torch.full(
            size=(
                self.cfg.n_channels,
                self.cfg.side_length,
                self.cfg.side_length,
            ),
            fill_value=self.cfg.background_intensity,
        )

        if y.item() == 0:
            x_base[
                :,
                shape_x : shape_x + self.square_len,
                shape_y : shape_y + self.square_len,
            ] = self.cfg.shape_intensity
        else:
            x_base = tensorcanvas.draw_circle(
                t=x_base,
                xp=shape_x + self.circle_rad,
                yp=shape_y + self.circle_rad,
                radius=self.circle_rad,
                color=torch.full(
                    size=(3,), fill_value=self.cfg.shape_intensity
                ),
            )

        x = (
            x_base
            + self.cfg.noise_std * torch.randn(size=x_base.shape, generator=rng)
        ).clip(min=0, max=1)

        return x, y

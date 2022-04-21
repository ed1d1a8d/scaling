from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random
from torchtyping import TensorType as T
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@dataclasses.dataclass
class HarmonicFnConfig:
    input_dim: int
    freq_limit: int
    num_components: int
    seed: int = 42


@typechecked
class HarmonicFn(pl.LightningModule):
    """
    Represents a bandlimited function with of a fixed number of
    integer-frequency harmonics.
    The function is scaled so as to be periodic on the unit-square.
    """

    def __init__(
        self,
        cfg: HarmonicFnConfig,
        coeffs: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        phases: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ):
        """
        Creates a harmonic function.
        Unspecified parameters are initialized randomly
        """
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
            )
            if coeffs is None
            else coeffs,
            requires_grad=requires_grad,
        )
        assert self.coeffs.shape == (cfg.num_components,)

        self.freqs = nn.Parameter(
            torch.randint(
                size=(cfg.num_components, cfg.input_dim),
                low=-cfg.freq_limit,
                high=cfg.freq_limit + 1,
                generator=rng,
            ).float()
            if freqs is None
            else freqs.float(),
            requires_grad=requires_grad,
        )
        assert self.freqs.shape == (cfg.num_components, cfg.input_dim)
        assert self.freqs.max() <= cfg.freq_limit

        self.phases = nn.Parameter(
            2
            * torch.pi
            * torch.rand(
                size=(cfg.num_components,),
                generator=rng,
            )
            if phases is None
            else phases,
            requires_grad=requires_grad,
        )
        assert self.phases.shape == (cfg.num_components,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        assert x.shape[-1] == self.cfg.input_dim

        # Could also do exp
        return (
            torch.cos(2 * torch.pi * x @ self.freqs.T + self.phases.reshape(1, -1))
            @ self.coeffs
        )

    def test_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log("test_mse", mse)

    def viz_2d(
        self,
        side_samples: int,
        pad: tuple[int, int] = (1, 5),
        value: float = 0.5,
    ) -> np.ndarray:
        assert sum(pad) + 2 == self.cfg.input_dim
        return utils.viz_2d(
            pred_fn=lambda xs: self.forward(
                F.pad(input=xs, pad=pad, mode="constant", value=value)
            ),
            side_samples=side_samples,
        )

    @staticmethod
    def harmonic_basis(
        x: T["batch":..., "D"],  # type: ignore
        freq_limit: int,
    ) -> tuple[T["batch":..., "basis_size"], T["num_freqs", "D"]]:  # type: ignore
        assert freq_limit >= 0
        D = x.shape[-1]

        def _include_freq(f: np.ndarray) -> bool:
            nonzero_coords = f[f != 0]
            if len(nonzero_coords) == 0:
                return True
            return nonzero_coords[0] > 0

        all_freqs = (
            np.mgrid[tuple(slice(-freq_limit, freq_limit + 1) for _ in range(D))]
            .reshape(D, -1)
            .T
        )
        freqs = torch.tensor(
            np.stack(
                [f for f in all_freqs if _include_freq(f)],
                axis=0,
            ),
            dtype=x.dtype,
            device=x.device,
        )

        prod = torch.einsum("...i, fi -> ...f", x, freqs)
        Phi = torch.cat(
            (
                torch.cos(2 * np.pi * prod),
                torch.sin(2 * np.pi * prod)[:, 1:],
            ),
            dim=-1,
        )
        Phi[:, 1:] *= np.sqrt(2)  # Orthonormalize basis

        return Phi, freqs

    @classmethod
    def construct_via_lstsq(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        freq_limit: int,
        lamb: float = 1e-4,
        coeff_threshold: float = 1e-6,
    ) -> HarmonicFn:
        assert ys.shape == xs.shape[:1]
        assert len(xs.shape) == 2
        assert freq_limit >= 0
        D = xs.shape[-1]

        Phi, freqs = cls.harmonic_basis(
            x=torch.tensor(xs),
            freq_limit=freq_limit,
        )
        Phi = Phi.numpy()
        freqs = freqs.numpy()
        num_freqs = freqs.shape[0]

        # all_coeffs = np.linalg.lstsq(a=Phi, b=ys, rcond=None)[0]
        # See https://stackoverflow.com/a/34171374/1337463
        all_coeffs = np.linalg.solve(
            a=Phi.T @ Phi + lamb * np.eye(Phi.shape[-1]),
            b=Phi.T @ ys,
        )
        all_coeffs[1:] *= np.sqrt(2)  # Change back to regular cos, sin basis

        cos_coeffs = all_coeffs[:num_freqs]
        sin_coeffs = np.pad(all_coeffs[num_freqs:], (1, 0))

        coeffs = np.sqrt(cos_coeffs ** 2 + sin_coeffs ** 2)
        phases = np.arctan2(-sin_coeffs, cos_coeffs)

        # cos(x + p) = cos(p) cos(x) - sin(p) sin(x)
        assert np.allclose(cos_coeffs, coeffs * np.cos(phases))
        assert np.allclose(sin_coeffs, -coeffs * np.sin(phases))

        sp_select = np.abs(coeffs) > coeff_threshold
        sparse_coeffs = coeffs[sp_select]
        sparse_freqs = freqs[sp_select]
        sparse_phases = phases[sp_select]

        return cls(
            cfg=HarmonicFnConfig(
                input_dim=D,
                freq_limit=int(sparse_freqs.max()),
                num_components=len(sparse_coeffs),
            ),
            coeffs=torch.tensor(sparse_coeffs).float(),
            freqs=torch.tensor(sparse_freqs, dtype=torch.long),
            phases=torch.tensor(sparse_phases).float(),
        )


@dataclasses.dataclass
class HarmonicFnTrainableConfig(HarmonicFnConfig):
    learning_rate: float = 3e-4
    sched_monitor: str = "train_mse"
    sched_patience: int = 25
    sched_decay: float = 0.1
    sched_min_lr: float = 1e-6

class HarmonicFnTrainable(HarmonicFn):
    """A HarmonicFn with trainable parameters."""

    def __init__(self, cfg: HarmonicFnTrainableConfig):
        super().__init__(cfg, requires_grad=True)

    def training_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)

        mse = F.mse_loss(input=y_hat, target=y)
        self.log("train_mse", mse)

        return mse

    def validation_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log("val_mse", mse)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        sched_config = dict(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt,
                mode="min",
                factor=self.cfg.sched_decay,
                patience=self.cfg.sched_patience,
                min_lr=self.cfg.sched_min_lr,
                verbose=True,
            ),
            monitor=self.cfg.sched_monitor,
        )

        return [opt], [sched_config]

from __future__ import annotations

import dataclasses
from typing import Optional

import einops
import numpy as np
import opt_einsum
import pytorch_lightning as pl
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random
from src.experiments.harmonics.harmonics import HarmonicFn
from torchtyping import TensorType as T
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@dataclasses.dataclass(frozen=True)
class ChebPolyConfig:
    input_dim: int
    deg_limit: int
    num_components: int
    seed: int = 42
    learning_rate: float = 3e-4

    # Whether to use a simpler but non-vectorized implementation of forward.
    simple_forward: bool = False


@typechecked
class ChebPoly(pl.LightningModule):
    """
    Represents a multivariate polynomial over [0, 1]^d that is a
    linear combination of Chebyshev polynomials.
    """

    def __init__(
        self,
        cfg: ChebPolyConfig,
        coeffs: Optional[torch.Tensor] = None,
        degs: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ):
        """
        Creates a linear combination of Chebyshev polynomials.
        Unspecified parameters are initialized randomly.
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

        self.degs = nn.Parameter(
            torch.randint(
                size=(cfg.num_components, cfg.input_dim),
                low=0,
                high=cfg.deg_limit + 1,
                generator=rng,
            ).long()
            if degs is None
            else degs.long(),
            requires_grad=False,
        )
        assert self.degs.shape == (cfg.num_components, cfg.input_dim)
        assert self.degs.max() <= cfg.deg_limit

    @staticmethod
    def cheb_basis(
        x: T["batch":..., "D"],  # type: ignore
        degs: T["basis_size", "D", int],  # type: ignore
        simple_mode: bool = False,
    ) -> T["batch":..., "basis_size"]:  # type: ignore
        deg_limit: int = degs.max().item() + 1

        cheb1d_ys = torch.ones((deg_limit + 1,) + x.shape)
        if deg_limit >= 1:
            cheb1d_ys[1] = 2 * x - 1
        for i in range(2, deg_limit + 1):
            cheb1d_ys[i] = 2 * (2 * x - 1) * cheb1d_ys[i - 1] - cheb1d_ys[i - 2]

        if simple_mode:
            basis_ys = []
            for deg in degs:
                cur_ys = torch.ones(x.shape[:-1])
                for i, di in enumerate(deg):
                    cur_ys *= cheb1d_ys[di, ..., i]
                basis_ys.append(cur_ys)

            return torch.stack(basis_ys, dim=-1)

        cheb1d_ys_degs = cheb1d_ys[degs]
        assert cheb1d_ys_degs.shape[: degs.ndim] == degs.shape
        assert cheb1d_ys_degs.shape[degs.ndim :] == cheb1d_ys.shape[1:]

        return einops.reduce(
            opt_einsum.contract("ij...j->ij...", cheb1d_ys_degs),
            "basis_idx input_dim ... -> ... basis_idx",
            reduction="prod",
        )

    def forward(self, x: T["batch":..., "input_dim"]) -> T["batch":...]:  # type: ignore
        assert x.shape[-1] == self.cfg.input_dim

        basis_ys: T["batch":..., "basis_size"] = self.cheb_basis(
            x=x,
            degs=self.degs,
            simple_mode=self.cfg.simple_forward,
        )

        return opt_einsum.contract("i, ...i -> ...", self.coeffs, basis_ys)

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

    @classmethod
    def construct_via_lstsq(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        deg_limit: int,
        freq_limit: int,
        hf_lambda: float,
        n_reg_samples: int = 512,
    ) -> ChebPoly:
        """
        xs are assumed to lie in [0, 1].
        """
        assert len(xs.shape) == 2
        assert ys.shape == xs.shape[:1]
        assert deg_limit >= 0
        assert freq_limit >= 0
        assert hf_lambda >= 0
        N, D = xs.shape

        ################################
        # Compute cheb_Phi
        ################################

        cheb_degs = (
            np.mgrid[tuple(slice(0, deg_limit + 1) for _ in range(D))].reshape(D, -1).T
        )
        with torch.no_grad():
            cheb_Phi = cls.cheb_basis(
                x=torch.tensor(xs),
                degs=torch.tensor(cheb_degs, dtype=torch.long),
            ).numpy()

        ################################
        # Compute fourier_reg_Phi and cheb_reg_Phi
        ################################

        xs_reg = np.random.uniform(low=0, high=1, size=(n_reg_samples, D))
        with torch.no_grad():
            fourier_reg_Phi = HarmonicFn.harmonic_basis(
                x=torch.tensor(xs_reg),
                freq_limit=freq_limit,
            )[0].numpy()
            assert fourier_reg_Phi.shape[0] == n_reg_samples
            assert len(fourier_reg_Phi.shape) == 2

            cheb_reg_Phi = cls.cheb_basis(
                x=torch.tensor(xs_reg),
                degs=torch.tensor(cheb_degs, dtype=torch.long),
            ).numpy()

        ################################
        # Compute cheb_coeffs
        ################################

        Q_reg = (
            fourier_reg_Phi @ np.linalg.pinv(fourier_reg_Phi) @ cheb_reg_Phi
            - cheb_reg_Phi
        )
        cheb_coeffs = np.linalg.solve(
            a=(cheb_Phi.T @ cheb_Phi + hf_lambda * Q_reg.T @ Q_reg),
            b=cheb_Phi.T @ ys,
        )

        return cls(
            cfg=ChebPolyConfig(
                input_dim=D,
                deg_limit=deg_limit,
                num_components=len(cheb_coeffs),
            ),
            coeffs=torch.tensor(cheb_coeffs, dtype=torch.float),
            degs=torch.tensor(cheb_degs, dtype=torch.long),
        )

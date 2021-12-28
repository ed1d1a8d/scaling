from __future__ import annotations

import dataclasses
from typing import Optional

import einops
import opt_einsum
import numpy as np
import pytorch_lightning as pl
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random


@dataclasses.dataclass(frozen=True)
class ChebPolyConfig:
    input_dim: int
    deg_limit: int
    num_components: int
    seed: int = 42
    learning_rate: float = 3e-4

    # Whether to use a simpler but non-vectorized implementation of forward.
    simple_forward: bool = False


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        assert x.shape[-1] == self.cfg.input_dim

        cheb1d_ys = torch.ones((self.cfg.deg_limit + 1,) + x.shape)
        if self.cfg.deg_limit >= 1:
            cheb1d_ys[1] = 2 * x - 1
        for i in range(2, self.cfg.deg_limit + 1):
            cheb1d_ys[i] = 2 * (2 * x - 1) * cheb1d_ys[i - 1] - cheb1d_ys[i - 2]

        if self.cfg.simple_forward:
            basis_ys = []
            for deg in self.degs:
                cur_ys = torch.ones(x.shape[:-1])
                for i, di in enumerate(deg):
                    cur_ys *= cheb1d_ys[di, ..., i]
                basis_ys.append(cur_ys)

            return opt_einsum.contract(
                "i, i... -> ...",
                self.coeffs,
                torch.stack(basis_ys, dim=0),
            )

        cheb1d_ys_degs = cheb1d_ys[self.degs]
        assert cheb1d_ys_degs.shape[0] == self.cfg.num_components
        assert cheb1d_ys_degs.shape[1] == self.cfg.input_dim
        assert cheb1d_ys_degs.shape[-1] == self.cfg.input_dim

        basis_ys = einops.reduce(
            opt_einsum.contract("ij...j->ij...", cheb1d_ys_degs),
            "basis_idx d ... -> basis_idx ...",
            reduction="prod",
        )

        return opt_einsum.contract("i, i... -> ...", self.coeffs, basis_ys)

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
    ):
        assert sum(pad) + 2 == self.cfg.input_dim
        utils.viz_2d(
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
    ) -> ChebPoly:
        assert len(xs.shape) == 2
        assert ys.shape == xs.shape[:1]
        assert deg_limit >= 0
        assert freq_limit >= 0
        assert hf_lambda >= 0
        N, D = xs.shape

        ################################
        # Compute cheb_Phi
        ################################

        cheb1d_ys = np.ones((deg_limit + 1,) + xs.shape)
        if deg_limit >= 1:
            cheb1d_ys[1] = 2 * xs - 1
        for i in range(2, deg_limit + 1):
            cheb1d_ys[i] = 2 * (2 * xs - 1) * cheb1d_ys[i - 1] - cheb1d_ys[i - 2]

        degs = (
            np.mgrid[tuple(slice(0, deg_limit + 1) for _ in range(D))].reshape(D, -1).T
        )
        cheb_J = len(degs)

        cheb1d_ys_degs = cheb1d_ys[degs]
        assert cheb1d_ys_degs.shape == (cheb_J, D, N, D)
        cheb_Phi = einops.reduce(
            opt_einsum.contract("ij...j->ij...", cheb1d_ys_degs),
            "basis_idx d ... -> basis_idx ...",
            reduction="prod",
        ).T
        assert cheb_Phi.shape == (N, cheb_J)

        ################################
        # Compute fourier_Phi
        ################################

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
        freqs = np.stack(
            [f for f in all_freqs if _include_freq(f)],
            axis=0,
        )

        fourier_Phi = np.concatenate(
            (
                np.cos(2 * np.pi * xs @ freqs.T),
                np.sin(2 * np.pi * xs @ freqs.T)[:, 1:],
            ),
            axis=-1,
        )

        # Orthonormalize basis
        # Not necessary, just makes things more numerically stable?
        fourier_Phi[:, 1:] *= np.sqrt(2)
        assert fourier_Phi.shape[0] == N
        assert len(fourier_Phi.shape) == 2

        ################################
        # Compute cheb_coeffs
        ################################

        # all_coeffs = np.linalg.lstsq(a=Phi, b=ys, rcond=None)[0]
        # See https://stackoverflow.com/a/34171374/1337463
        Q = fourier_Phi @ np.linalg.pinv(fourier_Phi) - np.eye(N)
        all_coeffs = np.linalg.solve(
            a=cheb_Phi.T @ (np.eye(N) + hf_lambda * Q.T @ Q) @ cheb_Phi,
            b=cheb_Phi.T @ ys,
        )

        return cls(
            cfg=ChebPolyConfig(
                input_dim=D,
                deg_limit=deg_limit,
                num_components=len(all_coeffs),
            ),
            coeffs=torch.Tensor(all_coeffs),
            degs=torch.Tensor(degs),
        )

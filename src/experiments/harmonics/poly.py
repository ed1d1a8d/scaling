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

        poly1d_ys = torch.ones((self.cfg.deg_limit + 1,) + x.shape)
        if self.cfg.deg_limit >= 1:
            poly1d_ys[1] = 2 * x - 1
        for i in range(2, self.cfg.deg_limit + 1):
            poly1d_ys[i] = 2 * (2 * x - 1) * poly1d_ys[i - 1] - poly1d_ys[i - 2]

        if self.cfg.simple_forward:
            basis_ys = []
            for deg in self.degs:
                cur_ys = torch.ones(x.shape[:-1])
                for i, di in enumerate(deg):
                    cur_ys *= poly1d_ys[di, ..., i]
                basis_ys.append(cur_ys)

            return opt_einsum.contract(
                "i, i... -> ...",
                self.coeffs,
                torch.stack(basis_ys, dim=0),
            )

        poly1d_ys_degs = poly1d_ys[self.degs]
        assert poly1d_ys_degs.shape[0] == self.cfg.num_components
        assert poly1d_ys_degs.shape[1] == self.cfg.input_dim
        assert poly1d_ys_degs.shape[-1] == self.cfg.input_dim

        basis_ys = einops.reduce(
            opt_einsum.contract("ij...j->ij...", poly1d_ys_degs),
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
        pass

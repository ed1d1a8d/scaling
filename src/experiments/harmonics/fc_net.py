import dataclasses
import enum
from typing import Optional

import mup
import numpy as np
import pytorch_lightning as pl
import src.experiments.harmonics.bw_loss as bw_loss
import src.utils as utils
import torch
import torch.nn.functional as F
from torch import nn, optim


class HFReg(enum.Enum):
    MCLS = enum.auto()
    DFT = enum.auto()
    NONE = enum.auto()


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ActivationT(enum.Enum):
    RELU = nn.ReLU
    SIN = SinActivation


@dataclasses.dataclass
class MinNormOptConfig:
    """
    Configuration for minnorm optimization,
    as described in https://arxiv.org/abs/1806.00730.

    Some hyperparemeters for MNIST are given in appendix D.
    """

    enabled: bool = False
    n_samples: Optional[int] = None

    # Better to be on the larger side? (see section 3.2.2)
    alpha_lr: float = 1

    # For stabilization, better to be small (see section 3.2.2)
    l2_reg_lambda: float = 1e-5

    def __post_init__(self):
        if self.enabled:
            assert self.n_samples is not None
            assert self.n_samples > 0


@dataclasses.dataclass
class FCNetConfig:
    input_dim: int = 2
    layer_widths: tuple[int, ...] = (96, 192, 1)
    act: ActivationT = ActivationT.RELU

    sparsity: float = 1
    output_sparsity: float = 0.5

    high_freq_reg: HFReg = HFReg.MCLS
    high_freq_lambda: float = 0
    high_freq_freq_limit: int = 0
    high_freq_mcls_samples: int = 1024
    high_freq_dft_ss: int = 8

    l0_clip: float = 1e-6

    l1_reg: bool = False
    l1_reg_lambda: float = 0
    l1_reg_lim: float = 0

    l2_reg: bool = False
    l2_reg_lambda: float = 0
    l2_reg_lim: float = 0  # l2_norm2

    minnorm_cfg: MinNormOptConfig = dataclasses.field(
        default_factory=lambda: MinNormOptConfig()
    )

    learning_rate: float = 1e-3
    sched_monitor: str = "train_loss"
    sched_patience: int = 25
    sched_decay: float = 0.1
    sched_min_lr: float = 1e-6
    sched_verbose: bool = True

    def __post_init__(self):
        assert 0 < self.sparsity <= 1


class FCNet(pl.LightningModule):
    """A fully connected neural network."""

    def __init__(
        self,
        cfg: FCNetConfig,
    ):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.layer_widths[-1] == 1

        _layers: list[nn.Module] = []
        if len(cfg.layer_widths) == 1:
            _layers.append(mup.MuReadout(cfg.input_dim, 1))
        else:
            _layers.append(nn.Linear(cfg.input_dim, cfg.layer_widths[0]))
            for w_in, w_out in zip(cfg.layer_widths[:-2], cfg.layer_widths[1:]):
                _layers.append(cfg.act.value())
                _layers.append(nn.Linear(w_in, w_out))
            _layers.append(cfg.act.value())
            _layers.append(mup.MuReadout(cfg.layer_widths[-2], 1))

        self.net = nn.Sequential(*_layers)
        mup.set_base_shapes(self.net, None)

        if self.cfg.sparsity < 1:
            with torch.no_grad():
                # Prune everything but output layer
                for param in self.net[:-1].parameters():
                    param.multiply_(
                        (torch.rand_like(param) <= self.cfg.sparsity)
                    )
                for param in self.net[-1].parameters():
                    param.multiply_(
                        (torch.rand_like(param) <= self.cfg.output_sparsity)
                    )

        if self.cfg.minnorm_cfg.enabled:
            self.alphas = nn.Parameter(
                torch.zeros(self.cfg.minnorm_cfg.n_samples)
            )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.squeeze(self.net(x), dim=-1)

    def get_l0_norm(self) -> torch.Tensor:
        return sum(
            (p.abs() > self.cfg.l0_clip).sum() for p in self.net.parameters()
        )

    def get_l1_norm(self) -> torch.Tensor:
        return sum(p.abs().sum() for p in self.net.parameters())

    def get_l2_norm2(self) -> torch.Tensor:
        return sum(p.square().sum() for p in self.net.parameters())

    def _minnorm_step_helper(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        log_prefix: str,
        optimizer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Based on algorithm 1 from https://arxiv.org/pdf/1806.00730.pdf.

        Our modifications:
            - We average instead of summing across the batch dimension, to
              match the behavior of mse_loss.
        """
        mn_cfg = self.cfg.minnorm_cfg

        x, y, idx = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log(f"{log_prefix}mse", mse)

        if optimizer_idx is None:
            return

        # Optimize weights
        if optimizer_idx == 0:
            l2_norm2 = self.get_l2_norm2()
            self.log(f"{log_prefix}l2_norm2", l2_norm2)

            # TODO: Also support minimum l1 norm solution
            loss_w = (
                l2_norm2
                + (self.alphas[idx] * (y - y_hat)).mean()
                + mn_cfg.l2_reg_lambda * mse
            )
            self.log(f"{log_prefix}loss_w", loss_w)
            return loss_w

        # Optimize alphas
        if optimizer_idx == 1:
            loss_a = -(self.alphas[idx] * (y - y_hat)).mean()
            self.log(f"{log_prefix}loss_a", loss_a)

            self.log(f"{log_prefix}alpha_avg", self.alphas.abs().mean())
            return loss_a

    def _step_helper(
        self,
        batch,
        log_prefix: str,
        optimizer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if self.cfg.minnorm_cfg.enabled:
            return self._minnorm_step_helper(
                batch=batch,
                log_prefix=log_prefix,
                optimizer_idx=optimizer_idx,
            )

        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log(f"{log_prefix}mse", mse)

        if self.cfg.high_freq_reg == HFReg.MCLS:
            hfn = bw_loss.high_freq_norm_mcls(
                fn=self.forward,
                input_dim=self.cfg.input_dim,
                freq_limit=self.cfg.high_freq_freq_limit,
                n_samples=self.cfg.high_freq_mcls_samples,
                device=torch.device(self.device),
            )
        elif self.cfg.high_freq_reg == HFReg.DFT:
            hfn = bw_loss.high_freq_norm_dft(
                fn=self.forward,
                input_dim=self.cfg.input_dim,
                freq_limit=self.cfg.high_freq_freq_limit,
                side_samples=self.cfg.high_freq_dft_ss,
                device=torch.device(self.device),
            )
        elif self.cfg.high_freq_reg == HFReg.NONE:
            hfn = 0
        else:
            raise ValueError(self.cfg.high_freq_reg)
        self.log(f"{log_prefix}hfn", hfn)

        with torch.no_grad():
            self.log(f"{log_prefix}l0_norm", self.get_l0_norm())

        l1_norm = self.get_l1_norm() if self.cfg.l1_reg else 0
        self.log(f"{log_prefix}l1_norm", l1_norm)

        l2_norm2 = self.get_l2_norm2() if self.cfg.l2_reg else 0
        self.log(f"{log_prefix}l2_norm2", l2_norm2)

        loss = (
            mse
            + self.cfg.high_freq_lambda * hfn
            + self.cfg.l1_reg_lambda * torch.relu(l1_norm - self.cfg.l1_reg_lim)
            + self.cfg.l2_reg_lambda
            * torch.relu(l2_norm2 - self.cfg.l2_reg_lim)
        )
        self.log(f"{log_prefix}loss", loss)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._step_helper(
            batch=batch, log_prefix="train_", optimizer_idx=optimizer_idx
        )

    def validation_step(self, batch, *_, **__):
        self._step_helper(batch=batch, log_prefix="val_")

    def test_step(self, batch, *_, **__):
        self._step_helper(batch=batch, log_prefix="test_")

    def configure_optimizers(self):
        opt = mup.MuAdam(self.net.parameters(), lr=self.cfg.learning_rate)
        sched_config = dict(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt,
                mode="min",
                factor=self.cfg.sched_decay,
                patience=self.cfg.sched_patience,
                min_lr=self.cfg.sched_min_lr,
                verbose=self.cfg.sched_verbose,
            ),
            monitor=self.cfg.sched_monitor,
        )

        if self.cfg.minnorm_cfg.enabled:
            opt_w = opt
            opt_a = optim.SGD([self.alphas], lr=self.cfg.minnorm_cfg.alpha_lr)
            return [opt_w, opt_a], [sched_config]
        else:
            return [opt], [sched_config]

    def viz_2d(
        self,
        side_samples: int,
        pad: tuple[int, int] = (0, 0),
        value: float = 0.5,
        lo: float = 0,
        hi: float = 1,
    ) -> np.ndarray:
        assert sum(pad) + 2 == self.cfg.input_dim
        return utils.viz_2d_hd(
            pred_fn=self.forward,
            side_samples=side_samples,
            pad=pad,
            value=value,
            lo=lo,
            hi=hi,
        )

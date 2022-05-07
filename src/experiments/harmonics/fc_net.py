import dataclasses
import enum

import numpy as np
import mup
import pytorch_lightning as pl
import src.experiments.harmonics.bw_loss as bw_loss
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import mup


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

    learning_rate: float = 1e-3
    sched_monitor: str = "train_mse"
    sched_patience: int = 25
    sched_decay: float = 0.1
    sched_min_lr: float = 1e-6
    sched_verbose: bool = False

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
                    param.multiply_((torch.rand_like(param) <= self.cfg.sparsity))
                for param in self.net[-1].parameters():
                    param.multiply_(
                        (torch.rand_like(param) <= self.cfg.output_sparsity)
                    )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.squeeze(self.net(x), dim=-1)

    def get_l0_norm(self) -> torch.Tensor:
        return sum((p.abs() > self.cfg.l0_clip).sum() for p in self.net.parameters())

    def get_l1_norm(self) -> torch.Tensor:
        return sum(p.abs().sum() for p in self.net.parameters())

    def get_l2_norm2(self) -> torch.Tensor:
        return sum(p.square().sum() for p in self.net.parameters())

    def _step_helper(self, batch, log_prefix: str) -> torch.Tensor:
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
            + self.cfg.l2_reg_lambda * torch.relu(l2_norm2 - self.cfg.l2_reg_lim)
        )
        self.log(f"{log_prefix}loss", loss)

        return loss

    def training_step(self, batch, *_, **__):
        return self._step_helper(batch=batch, log_prefix="train_")

    def validation_step(self, batch, *_, **__):
        self._step_helper(batch=batch, log_prefix="val_")

    def test_step(self, batch, *_, **__):
        self._step_helper(batch=batch, log_prefix="test_")

    def configure_optimizers(self):
        opt = mup.MuAdam(self.parameters(), lr=self.cfg.learning_rate)
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
        return utils.viz_2d(
            pred_fn=lambda xs: self.forward(
                F.pad(input=xs, pad=pad, mode="constant", value=value)
            ),
            side_samples=side_samples,
            lo=lo,
            hi=hi,
        )

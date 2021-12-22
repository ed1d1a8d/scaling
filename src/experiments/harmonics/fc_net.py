import dataclasses
import enum

import pytorch_lightning as pl
import src.experiments.harmonics.bw_loss as bw_loss
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class HFReg(enum.Enum):
    MCLS = "MCLS"
    DFT = "DFT"


@dataclasses.dataclass(frozen=True)
class FCNetConfig:
    input_dim: int = 2
    layer_widths: tuple[int, ...] = (96, 192, 1)

    high_freq_reg: HFReg = HFReg.MCLS
    high_freq_lambda: float = 0
    high_freq_bandlimit: int = 0
    high_freq_mcls_samples: int = 1024
    high_freq_dft_ss: int = 8

    learning_rate: float = 3e-4
    sched_monitor: str = "val_mse"
    sched_patience: int = 25
    sched_decay: float = 0.1
    sched_min_lr: float = 1e-6


class FCNet(pl.LightningModule):
    """A fully connected neural network."""

    def __init__(self, cfg: FCNetConfig):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.layer_widths[-1] == 1

        _layers = [nn.Linear(cfg.input_dim, cfg.layer_widths[0])]
        for w_in, w_out in zip(cfg.layer_widths, cfg.layer_widths[1:]):
            _layers.append(nn.ReLU())
            _layers.append(nn.Linear(w_in, w_out))

        self.net = nn.Sequential(*_layers)

        self.save_hyperparameters()

    def forward(self, x):
        return torch.squeeze(self.net(x), dim=-1)

    def _step_helper(self, batch, log_prefix: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log(f"{log_prefix}mse", mse)

        if self.cfg.high_freq_reg == HFReg.MCLS:
            hfn = bw_loss.high_freq_norm_mcls(
                fn=self.forward,
                input_dim=self.cfg.input_dim,
                bandlimit=self.cfg.high_freq_bandlimit,
                n_samples=self.cfg.high_freq_mcls_samples,
                device=self.device,
            )
        elif self.cfg.high_freq_reg == HFReg.DFT:
            hfn = bw_loss.high_freq_norm_dft(
                fn=self.forward,
                input_dim=self.cfg.input_dim,
                bandlimit=self.cfg.high_freq_bandlimit,
                side_samples=self.cfg.high_freq_dft_ss,
                device=self.device,
            )
        else:
            assert False
        self.log(f"{log_prefix}hfn", hfn)

        loss = mse + self.cfg.high_freq_lambda * hfn
        self.log(f"{log_prefix}loss", loss)

        return loss

    def training_step(self, batch, *_, **__):
        return self._step_helper(batch=batch, log_prefix="train_")

    def validation_step(self, batch, *_, **__):
        self._step_helper(batch=batch, log_prefix="val_")

    def test_step(self, batch, *_, **__):
        self._step_helper(batch=batch, log_prefix="test_")

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

    def viz_2d(
        self,
        side_samples: int,
        pad: tuple[int, int] = (0, 0),
        value: float = 0.5,
    ):
        assert sum(pad) + 2 == self.cfg.input_dim
        utils.viz_2d(
            pred_fn=lambda xs: self.forward(
                F.pad(input=xs, pad=pad, mode="constant", value=value)
            ),
            side_samples=side_samples,
        )

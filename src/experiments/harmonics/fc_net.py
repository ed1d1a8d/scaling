import dataclasses

import pytorch_lightning as pl
import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class FCNetConfig:
    input_dim: int = 2
    layer_widths: tuple[int, ...] = (96, 192, 1)
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

    def test_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log("test_mse", mse)

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

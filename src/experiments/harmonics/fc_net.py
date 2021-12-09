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

    def forward(self, x):
        return torch.squeeze(self.net(x), dim=-1)

    def training_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(input=y_hat, target=y)
        return loss

    def validation_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(input=y_hat, target=y)
        self.log("val_mse", loss)

    def test_step(self, batch, *_, **__):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(input=y_hat, target=y)
        self.log("test_mse", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)

    def viz_2d(self, side_samples: int):
        assert self.cfg.input_dim == 2
        utils.viz_2d(self.forward, side_samples=side_samples)

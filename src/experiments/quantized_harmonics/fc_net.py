import dataclasses
import enum

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torchtyping import TensorType as T
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@dataclasses.dataclass(frozen=True)
class FCNetConfig:
    input_resolution: int = 96
    input_dim: int = 2
    layer_widths: tuple[int, ...] = (128, 128, 128, 1)

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

        _layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(cfg.input_resolution * cfg.input_dim, cfg.layer_widths[0]),
        ]
        for w_in, w_out in zip(cfg.layer_widths, cfg.layer_widths[1:]):
            _layers.append(nn.ReLU())
            _layers.append(nn.Linear(w_in, w_out))

        self.net = nn.Sequential(*_layers)

        self.save_hyperparameters()

    @typechecked
    def forward(self, x: T["batch", "input_dim", int]) -> T["batch"]:
        x_one_hot: T["batch", "input_dim", "input_resolution", float] = F.one_hot(
            x,
            num_classes=self.cfg.input_resolution,
        ).float()
        return einops.rearrange(self.net(x_one_hot), "batch 1 -> batch")

    def _step_helper(self, batch, log_prefix: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(input=y_hat, target=y)
        self.log(f"{log_prefix}mse", mse)

        loss = mse
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
    ) -> np.ndarray:
        raise NotImplementedError

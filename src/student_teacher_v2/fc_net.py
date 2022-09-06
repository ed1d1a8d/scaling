from typing import Callable, Type

import mup
import numpy as np
import torch
from src.student_teacher_v2 import utils
from torch import nn


class FCNet(nn.Module):
    """A fully connected neural network."""

    def __init__(
        self,
        input_dim: int,
        layer_widths: tuple[int, ...],
        activation: Type[nn.Module] = nn.ReLU,
        end_with_activation: bool = False,
        zero_final_bias: bool = False,
        input_scale: float = 1.0,
        input_shift: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_widths = layer_widths

        self.input_scale = input_scale
        self.input_shift = input_shift

        _layers: list[nn.Module] = []
        if len(layer_widths) == 1:
            _layers.append(mup.MuReadout(input_dim, layer_widths[-1]))
        else:
            _layers.append(nn.Linear(input_dim, layer_widths[0]))
            for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:]):
                _layers.append(activation())
                _layers.append(nn.Linear(w_in, w_out))
            _layers.append(activation())
            _layers.append(mup.MuReadout(layer_widths[-2], layer_widths[-1]))

        if zero_final_bias:
            readout: mup.MuReadout = _layers[-1]  # type: ignore
            readout.bias.data.zero_()

        if end_with_activation:
            _layers.append(activation())

        self.net = nn.Sequential(*_layers)
        mup.set_base_shapes(self.net, None)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.net(self.input_scale * x + self.input_shift)

    def render_2d_slice(
        self,
        d1: int,
        d2: int,
        side_samples: int,
        batch_size: int,
        lo: float = 0,
        hi: float = 1,
        od_value: float = 0.5,
        output_transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ) -> np.ndarray:
        """Visualizes a 2d slice of a hypercube"""

        assert 0 <= d1 < self.input_dim
        assert 0 <= d2 < self.input_dim

        def custom_forward(xy_np: np.ndarray) -> np.ndarray:
            xy = torch.Tensor(xy_np)

            xy_full = torch.full(
                size=(len(xy), self.input_dim),
                fill_value=od_value,
            )
            xy_full[:, d1] = xy[:, 0]
            xy_full[:, d2] = xy[:, 1]

            with torch.no_grad():
                zs = utils.batch_execute(
                    fn=lambda xs: output_transform(
                        self.forward(xs.to(self.device))
                    ),
                    xs=xy_full,
                    batch_size=batch_size,
                )

            return zs.cpu().numpy()

        return utils.render_2d_image(
            fn=custom_forward, side_samples=side_samples, lo=lo, hi=hi
        )

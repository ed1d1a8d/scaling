import mup
import numpy as np
import src.utils as utils
import torch
from torch import nn


class FCNet(nn.Module):
    """A fully connected neural network."""

    def __init__(
        self,
        input_dim: int,
        layer_widths: tuple[int, ...],
    ):
        super().__init__()
        self.input_dim = input_dim

        _layers: list[nn.Module] = []
        if len(layer_widths) == 1:
            _layers.append(mup.MuReadout(input_dim, layer_widths[-1]))
        else:
            _layers.append(nn.Linear(input_dim, layer_widths[0]))
            for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:]):
                _layers.append(nn.ReLU())
                _layers.append(nn.Linear(w_in, w_out))
            _layers.append(nn.ReLU())
            _layers.append(mup.MuReadout(layer_widths[-2], layer_widths[-1]))

        self.net = nn.Sequential(*_layers)
        mup.set_base_shapes(self.net, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.net(x)

    def viz_2d(
        self,
        side_samples: int,
        pad: tuple[int, int] = (0, 0),
        value: float = 0.5,
        lo: float = 0,
        hi: float = 1,
    ) -> np.ndarray:
        assert sum(pad) + 2 == self.input_dim
        return utils.viz_2d_hd(
            pred_fn=lambda xs: self.forward(xs.cuda()).argmax(dim=-1).cpu(),
            side_samples=side_samples,
            pad=pad,
            value=value,
            lo=lo,
            hi=hi,
        )

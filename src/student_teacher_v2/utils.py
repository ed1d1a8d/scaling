from typing import Callable, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def ceil_div(a: int, b: int) -> int:
    """Returns ceil(a / b)"""
    return (a + b - 1) // b


def tag_dict(
    d: dict[str, T],
    prefix: str = "",
    suffix: str = "",
) -> dict[str, T]:
    return {f"{prefix}{key}{suffix}": val for key, val in d.items()}


def batch_execute(
    fn: Callable[[torch.Tensor], torch.Tensor],
    xs: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    for i in range(0, len(xs), batch_size):
        batch_xs = xs[i : i + batch_size]
        outs.append(fn(batch_xs))

    return torch.cat(outs)


def render_2d_image(
    fn: Callable[[np.ndarray], np.ndarray],
    side_samples: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Should only be called on a pred_fn that takes 2d inputs."""
    s = slice(0, 1, 1j * side_samples)

    XY = np.mgrid[s, s].T * (hi - lo) + lo
    assert XY.shape == (side_samples, side_samples, 2)

    zs = fn(XY.reshape(-1, 2))
    assert zs.shape == (side_samples**2, 1)

    return zs.reshape(side_samples, side_samples)

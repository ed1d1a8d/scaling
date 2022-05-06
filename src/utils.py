import pathlib
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_BASE = pathlib.Path(__file__).parent.parent.resolve()


def to_2d_image(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    side_samples: int,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Should only be called on a pred_fn that takes 2d inputs."""
    s = slice(0, 1, 1j * side_samples)
    XY = np.mgrid[s, s].T * (hi - lo) + lo
    img = pred_fn(torch.Tensor(XY))
    return img


def viz_2d(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    side_samples: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Should only be called on a pred_fn that takes 2d inputs."""
    with torch.no_grad():
        img = to_2d_image(pred_fn, side_samples, lo=lo, hi=hi).numpy()
    plt.imshow(img, origin="lower")
    return img


def plot_errorbar(
    xs: np.ndarray,
    ys: np.ndarray,
    lo_q: float = 0,
    mid_q: float = 0.5,
    hi_q: float = 1,
    **plt_kwargs
):
    lo = np.quantile(ys, lo_q, axis=-1)
    mid = np.quantile(ys, mid_q, axis=-1)
    hi = np.quantile(ys, hi_q, axis=-1)

    plt.errorbar(
        x=xs[mid != np.nan],
        y=mid[mid != np.nan],
        yerr=np.stack([mid - lo, hi - mid])[:, mid != np.nan],
        **plt_kwargs
    )

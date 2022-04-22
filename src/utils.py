import os
import pathlib
import pickle
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

REPO_BASE = pathlib.Path(__file__).parent.parent.resolve()


def mlflow_init():
    mlflow.set_tracking_uri(os.path.join(REPO_BASE, "mlruns"))


def mlflow_abs_path(
    run_id: str,
    rel_artifact_path: str,
):
    run = mlflow.get_run(run_id)
    return os.path.join(REPO_BASE, run.info.artifact_uri, rel_artifact_path)


def mlflow_read_pkl(
    run_id: str,
    rel_artifact_path: str,
):
    with open(
        mlflow_abs_path(run_id, rel_artifact_path),
        "rb",
    ) as f:
        return pickle.load(f)


def mlflow_read_img(
    run_id: str,
    rel_artifact_path: str,
) -> Image.Image:
    return Image.open(mlflow_abs_path(run_id, rel_artifact_path))


def to_2d_image(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    side_samples: int,
) -> torch.Tensor:
    """Should only be called on a pred_fn that takes 2d inputs."""
    s = slice(0, 1, 1j * side_samples)
    XY = np.mgrid[s, s].T
    img = pred_fn(torch.Tensor(XY))
    return img


def viz_2d(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    side_samples: int,
) -> np.ndarray:
    """Should only be called on a pred_fn that takes 2d inputs."""
    with torch.no_grad():
        img = to_2d_image(pred_fn, side_samples).numpy()
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

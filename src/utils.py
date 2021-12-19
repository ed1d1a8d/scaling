import os
import pathlib
import pickle
from typing import Callable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
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
):
    """Should only be called on a pred_fn that takes 2d inputs."""
    with torch.no_grad():
        img = to_2d_image(pred_fn, side_samples).numpy()
    plt.imshow(img, origin="lower")

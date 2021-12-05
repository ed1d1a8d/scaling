import os
import pathlib
import tempfile
from typing import Callable, Optional

import flax.serialization
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import msgpack

REPO_BASE = pathlib.Path(__file__).parent.parent.resolve()


def mlflow_log_jax(
    pytree,
    artifact_name: str,
    artifact_path: Optional[str] = None,
):
    b = flax.serialization.msgpack_serialize(pytree)
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, artifact_path or "", artifact_name)
        with open(fpath, "wb") as f:
            f.write(b)

        mlflow.log_artifacts(local_dir=td)


def msgpack_restore_v2(encoded_pytree: bytes):
    """
    flax.serialization._msgpack_ext_unpack with strict_map_key=False.
    See https://github.com/pupil-labs/pupil/issues/1830 for details.
    """
    state_dict = msgpack.unpackb(
        encoded_pytree,
        ext_hook=flax.serialization._msgpack_ext_unpack,
        raw=False,
        strict_map_key=False,
    )
    return flax.serialization._unchunk_array_leaves_in_place(state_dict)


def mlflow_abs_path(
    run_id: str,
    rel_artifact_path: str,
):
    run = mlflow.get_run(run_id)
    return os.path.join(REPO_BASE, run.info.artifact_uri, rel_artifact_path)


def mlflow_read_jax(
    run_id: str,
    rel_artifact_path: str,
):
    with open(
        mlflow_abs_path(run_id, rel_artifact_path),
        "rb",
    ) as f:
        b = f.read()
        return msgpack_restore_v2(b)


def mlflow_init():
    mlflow.set_tracking_uri(os.path.join(REPO_BASE, "mlruns"))


def to_2d_image(
    pred_fn: Callable[[jnp.ndarray], jnp.ndarray],
    side_samples: int,
) -> jnp.ndarray:
    """Should only be called on a pred_fn that takes 2d inputs."""
    s = slice(0, 1, 1j * side_samples)
    XY = jnp.mgrid[s, s].T
    img = pred_fn(XY)
    return img


def viz_2d(
    pred_fn: Callable[[jnp.ndarray], jnp.ndarray],
    side_samples: int,
):
    """Should only be called on a pred_fn that takes 2d inputs."""
    img = to_2d_image(pred_fn, side_samples)
    plt.imshow(img, origin="lower")

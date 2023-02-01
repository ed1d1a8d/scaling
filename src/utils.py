import contextlib
import datetime
import io
import json
import os
import pathlib
import tempfile
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn.functional as F
import wandb
import wandb.apis
from wandb.apis.internal import Api as InternalApi

REPO_BASE = pathlib.Path(__file__).parent.parent.resolve()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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


def viz_2d_hd(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    side_samples: int,
    pad: tuple[int, int] = (0, 0),
    value: float = 0.5,
    lo: float = 0,
    hi: float = 1,
) -> np.ndarray:
    """Can be called on a pred_fn that takes high dimensional inputs."""
    return viz_2d(
        pred_fn=lambda xs: pred_fn(
            F.pad(input=xs, pad=pad, mode="constant", value=value)
        ),
        side_samples=side_samples,
        lo=lo,
        hi=hi,
    )


def plot_errorbar(
    xs: np.ndarray,
    ys: np.ndarray,
    lo_q: float = 0,
    mid_q: float = 0.5,
    hi_q: float = 1,
    **plt_kwargs,
):
    lo = np.quantile(ys, lo_q, axis=-1)
    mid = np.quantile(ys, mid_q, axis=-1)
    hi = np.quantile(ys, hi_q, axis=-1)

    plt.errorbar(
        x=xs[mid != np.nan],
        y=mid[mid != np.nan],
        yerr=np.stack([mid - lo, hi - mid])[:, mid != np.nan],
        **plt_kwargs,
    )


def runs_to_df(runs: list[wandb.apis.public.Run]):
    def flatten_dict(d: dict, prefix: str = "") -> dict:
        ret = dict()
        for k, v in d.items():
            if isinstance(v, dict):
                ret |= flatten_dict(v, prefix=f"{k}_")
            else:
                ret[f"{prefix}{k}"] = v
        return ret

    return pd.DataFrame(
        [
            flatten_dict(r.summary._json_dict)  # type: ignore
            | flatten_dict(r.config)
            | {
                "id": r.id,
                "run_path": "/".join(r.path),
                "name": r.name,
                "state": r.state,
            }
            for r in runs
        ]
    )


def get_run_metadata(
    run: wandb.apis.public.Run,
    download_dir: Optional[str] = None,
) -> dict[str, Any]:
    wf: wandb.apis.public.File = run.file("wandb-metadata.json")

    if download_dir is None:
        # Create a temporary directory to download the artifact to
        with tempfile.TemporaryDirectory() as td:
            with wf.download(root=td) as file:
                return json.load(file)
    else:
        with wf.download(root=download_dir) as file:
            return json.load(file)


def get_run_command(
    run: wandb.apis.public.Run,
    download_dir: Optional[str] = None,
) -> list[str]:
    # https://docs.wandb.ai/guides/track/public-api-guide#get-the-command-that-ran-the-run
    meta = get_run_metadata(run, download_dir=download_dir)
    return ["python"] + [meta["program"]] + meta["args"]


def artifact_to_df(
    artifact: wandb.apis.public.Artifact,
    download_dir: Optional[str] = None,
) -> pd.DataFrame:
    if download_dir is None:
        # Create a temporary directory to download the artifact to
        with tempfile.TemporaryDirectory() as td:
            with open(artifact.file(root=td)) as file:
                json_dict = json.load(file)
    else:
        with open(artifact.file(root=download_dir)) as file:
            json_dict = json.load(file)

    return pd.DataFrame(
        json_dict["data"],
        columns=json_dict["columns"],
    )


def wandb_get_img(
    wf: wandb.apis.public.File,
    download_dir: Optional[str] = None,
) -> PIL.Image.Image:
    if download_dir is None:
        # Create a temporary directory to download the artifact to
        with tempfile.TemporaryDirectory() as td:
            with open(wf.download(root=td).name, "rb") as file:
                img = PIL.Image.open(file)
                img.load()
    else:
        with open(wf.download(root=download_dir).name, "rb") as file:
            img = PIL.Image.open(file)
            img.load()

    return img


def wandb_run_save_objs(
    run: wandb.apis.public.Run,
    img_dict: dict[str, Union[PIL.Image.Image, str]],
    extension: str = "png",
):
    """
    img_dict is a dict mapping wandb_file_paths to Images.
    wandb_file_paths are w.r.t. to the cloud wandb run storage.
    """
    iapi = InternalApi(
        default_settings={"entity": run.entity, "project": run.project},
        retry_timedelta=datetime.timedelta(seconds=20),
    )
    iapi.set_current_run_id(run.id)

    # See https://stackoverflow.com/a/49010489/1337463 for an ExitStack tutorial.
    with tempfile.TemporaryDirectory() as td:
        with contextlib.ExitStack() as stack:
            push_dict: dict[str, io.IOBase] = {}
            for idx, (wandb_file_path, obj) in enumerate(img_dict.items()):
                save_path = os.path.join(td, f"{idx}.png")

                if isinstance(obj, PIL.Image.Image):
                    assert wandb_file_path.endswith(extension)
                    obj.save(save_path, format=extension)
                else:
                    pathlib.Path(save_path).write_text(obj)

                push_dict[wandb_file_path] = stack.enter_context(
                    open(save_path, mode="rb")
                )

            iapi.push(push_dict)


def interactive_binary_query(x: Any) -> bool:
    """Interactively queries the user for a binary answer."""
    print("Answer for the object:", x)

    ans: bool = False
    while True:
        user_input = input("y/n? ").lower().strip()
        if user_input in ("y", "n"):
            ans = user_input == "y"
            break

    return ans


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def freeze(module: torch.nn.Module) -> None:
    """
    WARNING: Does not work for BatchNorm layers.
             In particular, we don't freeze the running mean and variance.
    """
    for param in module.parameters():
        param.requires_grad = False

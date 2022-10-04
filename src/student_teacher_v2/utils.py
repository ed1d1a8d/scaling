import dataclasses
import os
import warnings
from typing import Callable, Generic, Optional, TypeVar

import torch
import wandb
from torch import nn

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


def save_model(model: nn.Module, model_name: str):
    print("Saving model checkpoint...")
    torch.save(
        model.state_dict(),
        os.path.join(wandb.run.dir, f"{model_name}.ckpt"),  # type: ignore
    )
    print("Saved model checkpoint.")


def load_model(model: nn.Module, model_name: str):
    print("Loading model checkpoint...")
    path: str = os.path.join(wandb.run.dir, f"{model_name}.ckpt")  # type: ignore
    model.load_state_dict(torch.load(path))
    print("Loaded model checkpoint.")


@dataclasses.dataclass
class Metric(Generic[T]):
    data: T
    summary: Optional[str] = None


WANDB_METRIC_SUMMARY_MAP: dict[str, Optional[str]] = dict()


def wandb_log(d: dict[str, Metric]):
    for name, metric in d.items():
        if name not in WANDB_METRIC_SUMMARY_MAP:
            WANDB_METRIC_SUMMARY_MAP[name] = metric.summary
            if metric.summary is not None:
                wandb.define_metric(name=name, summary=metric.summary)
        elif WANDB_METRIC_SUMMARY_MAP[name] != metric.summary:
            s1 = WANDB_METRIC_SUMMARY_MAP[name]
            s2 = metric.summary
            warnings.warn(
                f"metric {name} has different summaries: {s1}, {s2}",
                RuntimeWarning,
            )

    wandb.log({name: metric.data for name, metric in d.items()})

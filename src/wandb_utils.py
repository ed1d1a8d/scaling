"""Custom wandb utilities."""
import dataclasses
import os
import warnings
from typing import Generic, Optional, TypeVar

import torch
import wandb
from torch import nn

T = TypeVar("T")


@dataclasses.dataclass
class WBMetric(Generic[T]):
    data: T
    summary: Optional[str] = None


class WandBManager:
    def __init__(self):
        self.metric_summary_map = dict()

    def log(self, d: dict[str, WBMetric]):
        for name, metric in d.items():
            if name not in self.metric_summary_map:
                self.metric_summary_map[name] = metric.summary
                if metric.summary is not None:
                    wandb.define_metric(name=name, summary=metric.summary)
            elif self.metric_summary_map[name] != metric.summary:
                s1 = self.metric_summary_map[name]
                s2 = metric.summary
                warnings.warn(
                    f"metric {name} has different summaries: {s1}, {s2}",
                    RuntimeWarning,
                )

        wandb.log({name: metric.data for name, metric in d.items()})


def tag_dict(
    d: dict[str, T],
    prefix: str = "",
    suffix: str = "",
) -> dict[str, T]:
    return {f"{prefix}{key}{suffix}": val for key, val in d.items()}


def save_model(model: nn.Module):
    """Save model to active wandb run directory."""
    print("Saving model checkpoint...")
    torch.save(
        model.state_dict(),
        os.path.join(wandb.run.dir, "model.ckpt"),  # type: ignore
    )
    print("Saved model checkpoint.")


def load_model(model: nn.Module):
    """Load model from active wandb run directory."""
    print("Loading model checkpoint...")
    path: str = os.path.join(wandb.run.dir, "model.ckpt")  # type: ignore
    model.load_state_dict(torch.load(path))
    print("Loaded model checkpoint.")

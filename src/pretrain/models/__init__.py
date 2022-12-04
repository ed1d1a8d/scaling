from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional

import PIL.Image
import torch
from torch import nn


@dataclasses.dataclass
class BaseEmbedderConfig(ABC):
    """Base class for embedder configurations."""

    id: str
    cache_dir: Optional[str] = "/home/gridsan/groups/ccg/huggingface/hub"

    def __post_init__(self):
        assert self.id in self.valid_model_ids

    @property
    @abstractmethod
    def valid_model_ids(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> BaseEmbedder:
        raise NotImplementedError


class BaseEmbedder(ABC, nn.Module):
    """
    Neural network that converts inputs into vector embeddings.
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_embedder_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

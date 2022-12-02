from __future__ import annotations

import dataclasses
from typing import Callable

import clip
import clip.model
import PIL.Image
import torch

from src.pretrain.models import BaseEmbedder, BaseEmbedderConfig


@dataclasses.dataclass
class OpenaiClipConfig(BaseEmbedderConfig):
    """Base class for model configurations."""

    id: str = "openai/ViT-B/32"

    @property
    def model_id(self) -> str:
        return "/".join(self.id.split("/")[1:])

    @property
    def valid_model_ids(self) -> tuple[str, ...]:
        return tuple(f"openai/{x}" for x in clip.available_models())

    def get_model(self) -> OpenaiClip:
        return OpenaiClip(self)


class OpenaiClip(BaseEmbedder):
    """
    Wrapper around the vision component of an OpenAI CLIP model.
    We use the original OpenAI implementation because it also supports
    both the ResNet and ViT architectures.
    """

    def __init__(self, cfg: OpenaiClipConfig):
        super().__init__()
        self.cfg = cfg

        self.clip: clip.model.CLIP
        self.preprocesser: Callable
        self.clip, self.preprocesser = clip.load(cfg.model_id)

    @property
    def name(self) -> str:
        return f"openai/{self.cfg.model_id}"

    @property
    def embed_dim(self) -> int:
        return self.clip.visual.output_dim

    @property
    def n_embedder_params(self) -> int:
        return sum(p.numel() for p in self.clip.visual.parameters())

    def preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
        return self.preprocesser(img)

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(xs)

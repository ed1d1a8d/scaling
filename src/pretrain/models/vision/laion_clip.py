from __future__ import annotations

import dataclasses

import torch
import PIL.Image
import transformers
from torch import nn

from src.pretrain.models.base import BaseEmbedder, BaseEmbedderConfig


@dataclasses.dataclass
class LaionClipConfig(BaseEmbedderConfig):
    """Base class for model configurations."""

    id: str = "hf/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    @property
    def hf_model_id(self) -> str:
        return "/".join(self.id.split("/")[1:])

    @property
    def valid_model_ids(self) -> tuple[str, ...]:
        # The commented out models don't work as of 2022-12-01.
        return (
            "hf/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            # "hf/laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
            # "hf/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
            "hf/laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
            # "hf/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
            "hf/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "hf/laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        )

    def get_model(self) -> LaionClip:
        return LaionClip(self)


class LaionClip(BaseEmbedder):
    """
    Wrapper around the vision component of a HuggingFace LAION CLIP model.
    """

    def __init__(self, cfg: LaionClipConfig):
        super().__init__()
        self.cfg = cfg

        laion_clip = transformers.AutoModel.from_pretrained(
            cfg.hf_model_id,
            cache_dir=cfg.cache_dir,
        )

        self.clip_vision_model: nn.Module = laion_clip.vision_model
        self._embed_dim: int = laion_clip.vision_embed_dim

        self.preprocesser: transformers.AutoProcessor
        self.preprocesser = transformers.AutoProcessor.from_pretrained(
            cfg.hf_model_id,
            cache_dir=cfg.cache_dir,
        )  # type: ignore

    @property
    def name(self) -> str:
        return self.cfg.hf_model_id

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def n_embedder_params(self) -> int:
        return sum(p.numel() for p in self.clip_vision_model.parameters())

    def preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
        d = self.preprocesser(
            images=img,
            return_tensors="pt",
        )  # type: ignore

        return d["pixel_values"][0]

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        return self.clip_vision_model(xs).pooler_output

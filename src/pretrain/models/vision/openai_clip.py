from __future__ import annotations

import dataclasses
from typing import Callable

import clip
import clip.model
import PIL.Image
import torch
from torch import nn

from src import utils
from src.pretrain.models.base import BaseEmbedder, BaseEmbedderConfig


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


class ParamWrapperModule(nn.Module):
    def __init__(self, *params: nn.parameter.Parameter):
        super().__init__()
        self.params = nn.ParameterList(params)


class OpenaiClip(BaseEmbedder):
    """
    Wrapper around the vision component of an OpenAI CLIP model.
    We use the original OpenAI implementation because it also supports
    both the ResNet and ViT architectures.
    """

    def __init__(self, cfg: OpenaiClipConfig):
        super().__init__()
        self.cfg = cfg

        clip_full: clip.model.CLIP
        self.preprocesser: Callable
        clip_full, self.preprocesser = clip.load(cfg.model_id)

        self.clip_visual = clip_full.visual

    @property
    def name(self) -> str:
        return f"openai/{self.cfg.model_id}"

    @property
    def embed_dim(self) -> int:
        return self.clip_visual.output_dim

    @property
    def n_embedder_params(self) -> int:
        return utils.count_params(self.clip_visual)

    @property
    def dtype(self):
        return self.clip_visual.conv1.weight.dtype

    def preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
        return self.preprocesser(img)

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        return self.clip_visual(xs.type(self.dtype))

    def get_layers_for_freezing(self) -> list[nn.Module]:
        """Internal function"""
        if isinstance(self.clip_visual, clip.model.VisionTransformer):
            # We try to break things up as much as possible to support fine-grained
            # freezing of the model.
            vit = self.clip_visual
            return [
                ParamWrapperModule(vit.class_embedding),
                ParamWrapperModule(vit.positional_embedding),
                vit.conv1,
                vit.ln_pre,
                *vit.transformer.resblocks,
                vit.ln_post,
                ParamWrapperModule(vit.proj),
            ]
        elif isinstance(self.clip_visual, clip.model.ModifiedResNet):
            raise NotImplementedError
        else:
            raise ValueError

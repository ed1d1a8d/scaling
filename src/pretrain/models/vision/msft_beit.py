from __future__ import annotations

import dataclasses

import PIL.Image
import torch
from transformers import AutoFeatureExtractor, BeitForImageClassification
from transformers.models.beit.modeling_beit import BeitModel

from src.pretrain.models.base import BaseEmbedder, BaseEmbedderConfig


@dataclasses.dataclass
class MsftBeitConfig(BaseEmbedderConfig):
    """Base class for model configurations."""

    id: str = "hf/microsoft/beit-base-patch16-224-pt22k"

    @property
    def hf_model_id(self) -> str:
        return "/".join(self.id.split("/")[1:])

    @property
    def valid_model_ids(self) -> tuple[str, ...]:
        return (
            "hf/microsoft/beit-base-finetuned-ade-640-640",
            "hf/microsoft/beit-base-patch16-224-pt22k-ft22k",
            "hf/microsoft/beit-base-patch16-224-pt22k",
            "hf/microsoft/beit-base-patch16-224",
            "hf/microsoft/beit-base-patch16-384",
            "hf/microsoft/beit-large-finetuned-ade-640-640",
            "hf/microsoft/beit-large-patch16-224-pt22k-ft22k",
            "hf/microsoft/beit-large-patch16-224-pt22k",
            "hf/microsoft/beit-large-patch16-224",
            "hf/microsoft/beit-large-patch16-384",
            "hf/microsoft/beit-large-patch16-512",
        )

    def get_model(self) -> MsftBeit:
        return MsftBeit(self)


class MsftBeit(BaseEmbedder):
    """
    Wrapper around the vision component of a HuggingFace Microsoft BEiT model.
    """

    def __init__(self, cfg: MsftBeitConfig):
        super().__init__()
        self.cfg = cfg

        self.beit: BeitModel = BeitForImageClassification.from_pretrained(  # type: ignore
            cfg.hf_model_id,
            cache_dir=cfg.cache_dir,
        ).beit  # type: ignore

        self.extractor = AutoFeatureExtractor.from_pretrained(
            cfg.hf_model_id,
            cache_dir=cfg.cache_dir,
        )

        with torch.no_grad():
            self.sample_input = self.preprocess(PIL.Image.new("RGB", (13, 19)))
            self._embed_dim: int = self.beit.forward(
                pixel_values=self.sample_input.unsqueeze(0)
            ).pooler_output.shape[  # type: ignore
                -1
            ]

    @property
    def name(self) -> str:
        return self.cfg.hf_model_id

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def n_embedder_params(self) -> int:
        return sum(p.numel() for p in self.beit.parameters())

    def preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
        d = self.extractor(
            images=img,
            return_tensors="pt",
        )  # type: ignore

        return d["pixel_values"][0]

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        return self.beit(pixel_values=xs).pooler_output  # type: ignore

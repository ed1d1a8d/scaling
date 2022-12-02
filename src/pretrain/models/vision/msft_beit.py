from __future__ import annotations

import dataclasses

import torch
from transformers import AutoFeatureExtractor, BeitForImageClassification
from transformers.models.beit.modeling_beit import BeitModel

from src.pretrain.models import BaseEmbedder, BaseEmbedderConfig


@dataclasses.dataclass
class MsftBeitConfig(BaseEmbedderConfig):
    """Base class for model configurations."""

    id: str = "hf/microsoft/beit-base-patch16-224-pt22k"

    @property
    def hf_model_id(self) -> str:
        return "/".join(id.split("/")[1:])

    @property
    def valid_model_ids(self) -> tuple[str, ...]:
        return (
            "microsoft/beit-base-finetuned-ade-640-640",
            "microsoft/beit-base-patch16-224-pt22k-ft22k",
            "microsoft/beit-base-patch16-224-pt22k",
            "microsoft/beit-base-patch16-224",
            "microsoft/beit-base-patch16-384",
            "microsoft/beit-large-finetuned-ade-640-640",
            "microsoft/beit-large-patch16-224-pt22k-ft22k",
            "microsoft/beit-large-patch16-224-pt22k",
            "microsoft/beit-large-patch16-224",
            "microsoft/beit-large-patch16-384",
            "microsoft/beit-large-patch16-512",
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
        ).beit  # type: ignore

        self.extractor = AutoFeatureExtractor.from_pretrained(cfg.hf_model_id)

        with torch.no_grad():
            self.sample_input = self.preprocess(torch.zeros(1, 10, 10, 3))
            self._embed_dim: int = self.beit.forward(
                pixel_values=self.sample_input
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

    def preprocess(self, xs: torch.Tensor) -> torch.Tensor:
        d = self.extractor(
            images=[x for x in xs],
            return_tensors="pt",
            batched=True,
        )  # type: ignore

        return d["pixel_values"]

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        return self.beit(xs).pooler_output  # type: ignore

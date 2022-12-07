from __future__ import annotations

import dataclasses

import torch
import PIL.Image
import transformers
from torch import nn

from src.pretrain.models import BaseEmbedder, BaseEmbedderConfig


@dataclasses.dataclass
class BertConfig(BaseEmbedderConfig):
    """Base class for model configurations."""

    id: str = "hf/bert-base-uncased"

    @property
    def hf_model_id(self) -> str:
        return "/".join(self.id.split("/")[1:])

    @property
    def valid_model_ids(self) -> tuple[str, ...]:
        # The commented out models don't work as of 2022-12-01.
        return (
            "hf/bert-base-uncased",
            "hf/bert-base-cased",
        )

    def get_model(self) -> Bert:
        return Bert(self)


class Bert(BaseEmbedder):
    """
    Wrapper around the vision component of a HuggingFace Bert model.
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg

        bert = transformers.BertModel.from_pretrained(
            cfg.hf_model_id,
            cache_dir=cfg.cache_dir,
        )

        self.model = bert

        # TODO not sure how to find the equivalent of vision_embed_dim for BERT
        self._embed_dim: int = bert.vision_embed_dim

        self.tokenizer: transformers.BertTokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
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
        return sum(p.numel() for p in self.model.parameters())

    def preprocess(self, text: str) -> torch.Tensor:
        d = self.tokenizer(
            text,
            return_tensors="pt",
        )  # type: ignore

        return d

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        return self.bert_language_model(xs).pooler_output

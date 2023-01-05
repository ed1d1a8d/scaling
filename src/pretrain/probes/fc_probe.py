"""
A fully connected probe on top of an embedding model.
"""
from __future__ import annotations

import dataclasses

import torch
from torch import nn

from src.pretrain.models.base import BaseEmbedder


@dataclasses.dataclass
class FCProbeConfig:
    n_layers: int = 2
    hidden_dim: int = 1024
    n_classes: int = 10

    def get_fc_probe(self, embedder: BaseEmbedder) -> FCProbe:
        return FCProbe(cfg=self, embedder=embedder)


class FCProbe(nn.Module):
    def __init__(
        self,
        cfg: FCProbeConfig,
        embedder: BaseEmbedder,
    ):
        super().__init__()
        self.cfg = cfg
        self.embedder = embedder

        layers: list[nn.Module] = []
        if cfg.n_layers == 1:
            layers.append(nn.Linear(embedder.embed_dim, cfg.n_classes))
        else:
            layers.append(nn.Linear(embedder.embed_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(cfg.n_layers - 2):
                layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(cfg.hidden_dim, cfg.n_classes))

        self.probe = nn.Sequential(*layers)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder.get_embeddings(xs)
        return self.probe(embeddings)

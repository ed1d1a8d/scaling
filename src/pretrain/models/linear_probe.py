import torch
from torch import nn

from src.pretrain.models import BaseEmbedder


class LinearProbe(nn.Module):
    def __init__(self, embedder: BaseEmbedder, n_classes: int):
        super().__init__()
        self.embedder = embedder
        self.probe = nn.Linear(embedder.embed_dim, n_classes)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder.get_embeddings(xs)
        return self.probe(embeddings)

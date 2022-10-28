from typing import Callable

import clip
import clip.model
import torch
from torch import nn


class CLIPClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.clip: clip.model.CLIP
        self.preprocess: Callable[[torch.Tensor], torch.Tensor]
        self.clip, self.preprocess = clip.load(model_name, device="cuda")

        self.embed_dim: int = self.clip.visual.output_dim

        self.clf_head = nn.Linear(
            self.embed_dim, self.num_classes, device="cuda"
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        embeddings = self.clip.encode_image(imgs)
        logits = self.clf_head(embeddings)
        return logits

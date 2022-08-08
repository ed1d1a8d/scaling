"""
Adapted from https://github.com/omihub777/ViT-CIFAR.
Augmented with https://github.com/microsoft/mup.
"""

from typing import Any, Union

import mup
import torch
import torch.nn.functional as F
from torch import nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.head = head
        self.feats = feats

        # self.sqrt_d = self.feats**0.5
        # We divide by 19.5 so d ~= sqrt_d when feats == 384
        self.d = self.feats / 19.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = (
            self.q(x)
            .view(b, n, self.head, self.feats // self.head)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(b, n, self.head, self.feats // self.head)
            .transpose(1, 2)
        )
        v = (
            self.v(x)
            .view(b, n, self.head, self.feats // self.head)
            .transpose(1, 2)
        )

        score = F.softmax(
            torch.einsum("bhif, bhjf->bhij", q, k) / self.d, dim=-1
        )  # (b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)  # (b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class TransformerEncoder(nn.Module):
    def __init__(
        self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0
    ):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        num_input_channels: int = 3,
        num_classes: int = 10,
        patch: int = 8,
        dropout: float = 0.0,
        num_layers: int = 7,
        hidden: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        is_cls_token: bool = True,
        mean: Union[tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[tuple[float, ...], float] = CIFAR10_STD,
    ):
        super().__init__()
        self.mean = nn.parameter.Parameter(
            torch.tensor(mean).view(num_input_channels, 1, 1)
        )
        self.std = nn.parameter.Parameter(
            torch.tensor(std).view(num_input_channels, 1, 1)
        )

        self.patch = patch  # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3  # 48 # patch vec length
        num_tokens = (
            (self.patch**2) + 1 if self.is_cls_token else (self.patch**2)
        )

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        enc_list = [
            TransformerEncoder(
                hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head
            )
            for _ in range(num_layers)
        ]
        self.enc = nn.Sequential(*enc_list)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            # nn.Linear(hidden, num_classes),  # for cls_token
            mup.MuReadout(hidden, num_classes),
        )

    def forward(self, x):
        out = (x - self.mean) / self.std

        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat(
                [self.cls_token.repeat(out.size(0), 1, 1), out], dim=1
            )
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)

        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


def get_mup_vit(**kwargs) -> ViT:
    def rm_scaled_dims(d: dict[str, Any]):
        return {
            k: v
            for k, v in d.items()
            if k not in ("head", "hidden", "mlp_hidden")
        }

    base_model = ViT(
        head=6, hidden=384, mlp_hidden=384, **rm_scaled_dims(kwargs)
    ).to("meta")
    delta_model = ViT(
        head=12, hidden=384 * 2, mlp_hidden=384 * 2, **rm_scaled_dims(kwargs)
    ).to("meta")

    model = ViT(**kwargs)
    mup.set_base_shapes(model, base_model, delta=delta_model)

    return model

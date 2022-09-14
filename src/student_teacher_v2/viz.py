"""Visualization utilities"""
import contextlib
from typing import Callable

import einops
import numpy as np
import PIL.Image
import torch
import torchvision
import wandb
from src.student_teacher_v2.fc_net import FCNet


def render_2d_image(
    fn: Callable[[np.ndarray], np.ndarray],
    side_samples: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Should only be called on a pred_fn that takes 2d inputs."""
    s = slice(0, 1, 1j * side_samples)

    XY = einops.rearrange(
        np.mgrid[s, s] * (hi - lo) + lo, "d h w -> h w d", d=2
    )
    assert XY.shape == (side_samples, side_samples, 2)

    zs = fn(XY.reshape(-1, 2))
    assert zs.shape == (side_samples**2, 1)

    return zs.reshape(side_samples, side_samples)


def get_student_teacher_viz_reg(
    student_net: FCNet,
    teacher_net: FCNet,
    n_viz_dims: int,
    viz_side_samples: int,
    input_lo: float,
    input_hi: float,
    od_value: float,
    batch_size: int,
    precision_context: contextlib.AbstractContextManager,
) -> list[wandb.Image]:
    assert student_net.input_dim == teacher_net.input_dim
    assert student_net.layer_widths[-1] == teacher_net.layer_widths[-1] == 1

    def get_render(net: FCNet, d1: int, d2: int) -> torch.Tensor:
        with precision_context:
            raw_render = net.render_2d_slice(
                d1=d1,
                d2=d2,
                side_samples=viz_side_samples,
                batch_size=batch_size,
                lo=input_lo,
                hi=input_hi,
                od_value=od_value,
            )
        return torch.tensor(raw_render[np.newaxis, :, :]).float()

    imgs: list[wandb.Image] = []
    n_viz_dims = min(n_viz_dims, student_net.input_dim)
    for d1 in range(n_viz_dims):
        for d2 in range(d1 + 1, n_viz_dims):
            student_render = get_render(student_net, d1=d1, d2=d2)
            teacher_render = get_render(teacher_net, d1=d1, d2=d2)

            mean = teacher_render.mean()
            std = torch.std(teacher_render) + 1e-6

            student_render = ((student_render - mean) / std).sigmoid()
            teacher_render = ((teacher_render - mean) / std).sigmoid()

            combined_render: np.ndarray = (
                torchvision.utils.make_grid(
                    [student_render, teacher_render], nrow=2, ncol=1
                )
                .cpu()
                .numpy()
            )[0]
            assert combined_render.ndim == 2

            combined_img = PIL.Image.fromarray(np.uint8(combined_render * 255))

            imgs.append(
                wandb.Image(
                    combined_img,
                    caption=f"Student (left), Teacher (right); h={d1} w={d2}",
                )
            )

    return imgs


def get_student_teacher_viz_clf(
    student_net: FCNet,
    teacher_net: FCNet,
    n_viz_dims: int,
    viz_side_samples: int,
    input_lo: float,
    input_hi: float,
    od_value: float,
    batch_size: int,
    precision_context: contextlib.AbstractContextManager,
) -> list[wandb.Image]:
    assert student_net.input_dim == teacher_net.input_dim
    assert student_net.layer_widths[-1] == teacher_net.layer_widths[-1]

    n_classes = student_net.layer_widths[-1]
    assert n_classes >= 2

    def get_render(net: FCNet, d1: int, d2: int, soft: bool) -> torch.Tensor:
        with precision_context:
            raw_render = net.render_2d_slice(
                d1=d1,
                d2=d2,
                side_samples=viz_side_samples,
                batch_size=batch_size,
                lo=input_lo,
                hi=input_hi,
                od_value=od_value,
                output_transform=lambda x: (
                    (
                        x.softmax(dim=-1)
                        @ torch.linspace(
                            start=0, end=1, steps=n_classes, device=net.device
                        )
                    ).unsqueeze(-1)
                    if soft
                    else x.argmax(dim=-1, keepdim=True) / (n_classes - 1)
                ),
            )
        return torch.tensor(raw_render[np.newaxis, :, :]).float()

    imgs: list[wandb.Image] = []
    n_viz_dims = min(n_viz_dims, student_net.input_dim)
    for d1 in range(n_viz_dims):
        for d2 in range(d1 + 1, n_viz_dims):
            student_render_hard = get_render(
                student_net, d1=d1, d2=d2, soft=False
            )
            teacher_render_hard = get_render(
                teacher_net, d1=d1, d2=d2, soft=False
            )

            student_render_soft = get_render(
                student_net, d1=d1, d2=d2, soft=True
            )
            teacher_render_soft = get_render(
                teacher_net, d1=d1, d2=d2, soft=True
            )

            combined_render: np.ndarray = (
                torchvision.utils.make_grid(
                    [
                        student_render_soft,
                        teacher_render_soft,
                        student_render_hard,
                        teacher_render_hard,
                    ],
                    nrow=2,
                    ncol=2,
                )
                .cpu()
                .numpy()
            )[0]
            assert combined_render.ndim == 2

            combined_img = PIL.Image.fromarray(np.uint8(combined_render * 255))

            imgs.append(
                wandb.Image(
                    combined_img,
                    caption="\n".join(
                        [
                            f"h={d1} w={d2}",
                            "student soft (top-left), teacher soft (top-right)",
                            "student hard (bot-left), teacher hard (bot-right)",
                        ],
                    ),
                )
            )

    return imgs

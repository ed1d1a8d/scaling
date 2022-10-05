import os
from typing import Optional, Sequence

import ffcv.transforms as transforms
import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from torch import nn

MNIST_ROOT = "/home/gridsan/groups/ccg/data/scaling/mnist20m"
MNIST_ROOT_ORIG = "/home/gridsan/groups/ccg/data/scaling/mnist-orig"


class Reshape(nn.Module):
    def __init__(self, target_shape: tuple[int]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(self.target_shape)


def cls_name(idx: int) -> str:
    return [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ][idx]


def get_loader(
    orig: bool,
    split: str,
    batch_size: int,
    indices: Optional[Sequence[int]] = None,
    num_workers: int = 20,
    random_order: bool = False,
    os_cache: bool = True,
    seed: int = 42,
) -> Loader:
    """
    Returns a loader for the first n elements of the split.
    If n is none, returns the entire split.
    """
    device = f"cuda:{torch.cuda.current_device()}"

    label_pipeline = [
        IntDecoder(),
        transforms.ToTensor(),
        transforms.ToDevice(device),
        transforms.Squeeze(),
    ]

    image_pipeline = [
        NDArrayDecoder(),
        transforms.ToTensor(),
        transforms.ToDevice(device),
        transforms.Convert(torch.float16),
        Reshape((-1, 1, 28, 28)),
        torchvision.transforms.Normalize(
            mean=0, std=255
        ),  # Images transformed to [0, 1] range.
    ]

    ROOT = MNIST_ROOT_ORIG if orig else MNIST_ROOT

    loader = Loader(
        os.path.join(ROOT, f"{split}.beton"),
        batch_size=batch_size,
        num_workers=num_workers,
        os_cache=os_cache,
        order=OrderOption.RANDOM if random_order else OrderOption.SEQUENTIAL,  # type: ignore
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        seed=seed,
        indices=indices,  # type: ignore
    )
    assert (indices is None) or (max(indices) < loader.reader.num_samples)

    return loader

import os
from typing import Optional, Sequence

import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Convert, Squeeze, ToDevice, ToTensor, ToTorchImage
import torch.distributed as dist

CIFAR_ROOT = "/home/gridsan/groups/ccg/data/scaling/cifar5m"


def cls_name(idx: int) -> str:
    return [
        "plane",  # "airplane",
        "car",  # "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ][idx]


def get_loader(
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
    device = f"cuda:{dist.get_rank()}"

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(device),
        Squeeze(),
    ]

    image_pipeline = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(device),
        ToTorchImage(channels_last=True),
        Convert(torch.float16),
        torchvision.transforms.Normalize(
            mean=0, std=255
        ),  # Images transformed to [0, 1] range.
    ]

    loader = Loader(
        os.path.join(CIFAR_ROOT, f"{split}.beton"),
        batch_size=batch_size,
        num_workers=num_workers,
        os_cache=os_cache,
        order=OrderOption.RANDOM if random_order else OrderOption.SEQUENTIAL,  # type: ignore
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        seed=seed,
        indices=indices,  # type: ignore
        distributed=True
    )
    assert (indices is None) or (max(indices) < loader.reader.num_samples)

    return loader

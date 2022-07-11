import os
from typing import Optional

import git.repo
import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Convert, Squeeze, ToDevice, ToTensor, ToTorchImage

GIT_ROOT = str(
    git.repo.Repo(".", search_parent_directories=True).working_tree_dir
)


def get_loader(
    split: str,
    batch_size: int,
    n: Optional[int] = None,
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
        os.path.join(GIT_ROOT, f"data/cifar5m/{split}.beton"),
        batch_size=batch_size,
        num_workers=num_workers,
        os_cache=os_cache,
        order=OrderOption.RANDOM if random_order else OrderOption.SEQUENTIAL,  # type: ignore
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        seed=seed,
        indices=None if n is None else range(n),  # type: ignore
    )
    assert (n is None) or (n <= loader.reader.num_samples)

    return loader

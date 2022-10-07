"""Code adapted from https://github.com/openai/CLIP"""

import os
import pathlib

import clip
import git.repo
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

GIT_ROOT = pathlib.Path(
    str(git.repo.Repo(".", search_parent_directories=True).working_tree_dir)
)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device)

# Load the dataset
# TODO: Handle more datasets besides CIFAR10
root = os.path.expanduser("/var/tmp/scratch")
train = CIFAR10(root, download=True, train=True, transform=preprocess)
test = CIFAR10(root, download=True, train=False, transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=256)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return (
        torch.cat(all_features).cpu().numpy(),
        torch.cat(all_labels).cpu().numpy(),
    )


# Calculate the image features
xs_train, ys_train = get_features(train)
xs_test, ys_test = get_features(test)

np.savez(
    GIT_ROOT / "data/clip-embeddings/cifar10_clip.npz",
    xs_train=xs_train,
    ys_train=ys_train,
    xs_test=xs_test,
    ys_test=ys_test,
)

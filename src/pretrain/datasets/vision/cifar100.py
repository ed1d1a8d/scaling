import dataclasses
from typing import Callable, Optional

import torch.utils.data
import torchvision

from src.pretrain.datasets.base import BaseDatasetConfig


@dataclasses.dataclass
class CIFAR100(BaseDatasetConfig):
    id: str = "cifar100"

    # TODO: Add support for configuring fine vs coarse labels.

    @property
    def text_from_website(self) -> tuple[str, ...]:
        return (
            "aquatic mammals: beaver, dolphin, otter, seal, whale",
            "fish: aquarium fish, flatfish, ray, shark, trout",
            "flowers: orchids, poppies, roses, sunflowers, tulips",
            "food containers: bottles, bowls, cans, cups, plates",
            "fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers",
            "household electrical devices: clock, computer keyboard, lamp, telephone, television",
            "household furniture: bed, chair, couch, table, wardrobe",
            "insects: bee, beetle, butterfly, caterpillar, cockroach",
            "large carnivores: bear, leopard, lion, tiger, wolf",
            "large man-made outdoor things: bridge, castle, house, road, skyscraper",
            "large natural outdoor scenes: cloud, forest, mountain, plain, sea",
            "large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo",
            "medium-sized mammals: fox, porcupine, possum, raccoon, skunk",
            "non-insect invertebrates: crab, lobster, snail, spider, worm",
            "people: baby, boy, girl, man, woman",
            "reptiles: crocodile, dinosaur, lizard, snake, turtle",
            "small mammals: hamster, mouse, rabbit, shrew, squirrel",
            "trees: maple, oak, palm, pine, willow",
            "vehicles 1: bicycle, bus, motorcycle, pickup truck, train",
            "vehicles 2: lawn-mower, rocket, streetcar, tank, tractor",
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(
            cn
            for line in self.text_from_website
            for cn in line.split(": ")[1].split(", ")
        )

    def superclass_names(self) -> tuple[str, ...]:
        return tuple(line.split(": ")[0] for line in self.text_from_website)

    def get_train_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        return torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )

    def get_test_ds(
        self, transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        return torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform,
        )

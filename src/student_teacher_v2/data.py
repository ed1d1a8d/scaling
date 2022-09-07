from typing import Callable

import torch
from src.student_teacher_v2 import utils


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    From https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6.
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        device: str,
        seed: int = 0,
    ):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.seed = seed

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tuple(t.to(self.device) for t in tensors)

        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)

    @property
    def dataset_len(self):
        return self.tensors[0].shape[0]

    @property
    def n_batches(self):
        return utils.ceil_div(self.dataset_len, self.batch_size)

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len, generator=self.rng)
            self.tensors = tuple(t[r] for t in self.tensors)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(
            t[self.i : self.i + self.batch_size] for t in self.tensors
        )
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class InfiniteTensorDataLoader:
    def __init__(
        self,
        gen_batch_fn: Callable[
            [int, torch.Generator], tuple[torch.Tensor, ...]
        ],
        batch_size: int,
        device: str,
        seed: int = 0,
    ):
        self.gen_batch_fn = gen_batch_fn
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self.rng = torch.Generator().manual_seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.gen_batch_fn(self.batch_size, self.rng)

    def __len__(self):
        raise TypeError("Tried to get length of infinite dataloader")

from __future__ import annotations

import dataclasses
import pickle
from typing import Optional, Sequence

import numpy as np


def subsample(
    xs: np.ndarray,
    ys: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsamples data."""
    assert len(xs) == len(ys)
    assert n_samples <= len(xs)

    sampled_indices = rng.choice(len(xs), n_samples, replace=False)

    return xs[sampled_indices], ys[sampled_indices]


def subsample_per_class(
    xs: np.ndarray,
    ys: np.ndarray,
    n_samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsamples data indepdently per class."""
    assert len(xs) == len(ys)

    sampled_indices = []
    for y in np.unique(ys):
        indices = np.where(ys == y)[0]
        sampled_indices.extend(
            rng.choice(indices, n_samples_per_class, replace=False)
        )

    return xs[sampled_indices], ys[sampled_indices]


@dataclasses.dataclass
class EmbeddingDataset:
    xs_train: np.ndarray
    ys_train: np.ndarray
    xs_test: np.ndarray
    ys_test: np.ndarray

    dataset_id: str
    embedder_id: str
    n_embedder_params: int = 0

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.ys_train))

    @property
    def min_samples_per_class(self) -> int:
        return min(np.sum(self.ys_train == y) for y in np.unique(self.ys_train))

    def save_to_file(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dataclasses.asdict(self), f)

    @classmethod
    def load_from_file(cls, path: str):
        with open(path, "rb") as f:
            dict_ = pickle.load(f)
        return cls(**dict_)

    def astype(self, dtype: np.dtype) -> EmbeddingDataset:
        return dataclasses.replace(
            self,
            xs_train=self.xs_train.astype(dtype),
            ys_train=self.ys_train.astype(dtype),
            xs_test=self.xs_test.astype(dtype),
            ys_test=self.ys_test.astype(dtype),
        )

    def filter_classes(self, classes: Sequence[int]) -> EmbeddingDataset:
        mask_train = np.isin(self.ys_train, classes)
        mask_test = np.isin(self.ys_test, classes)
        return dataclasses.replace(
            self,
            xs_train=self.xs_train[mask_train],
            ys_train=self.ys_train[mask_train],
            xs_test=self.xs_test[mask_test],
            ys_test=self.ys_test[mask_test],
        )

    def subsample(
        self,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> EmbeddingDataset:
        rng = np.random.default_rng() if rng is None else rng

        xs_train, ys_train = (
            subsample(self.xs_train, self.ys_train, n_train, rng)
            if n_train is not None
            else (self.xs_train, self.ys_train)
        )

        xs_test, ys_test = (
            subsample(self.xs_test, self.ys_test, n_test, rng)
            if n_test is not None
            else (self.xs_test, self.ys_test)
        )

        return dataclasses.replace(
            self,
            xs_train=xs_train,
            ys_train=ys_train,
            xs_test=xs_test,
            ys_test=ys_test,
        )

    def subsample_per_class(
        self,
        n_train_per_class: Optional[int] = None,
        n_test_per_class: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> EmbeddingDataset:
        rng = np.random.default_rng() if rng is None else rng

        xs_train, ys_train = (
            subsample_per_class(
                self.xs_train, self.ys_train, n_train_per_class, rng
            )
            if n_train_per_class is not None
            else (self.xs_train, self.ys_train)
        )

        xs_test, ys_test = (
            subsample_per_class(
                self.xs_test, self.ys_test, n_test_per_class, rng
            )
            if n_test_per_class is not None
            else (self.xs_test, self.ys_test)
        )

        return dataclasses.replace(
            self,
            xs_train=xs_train,
            ys_train=ys_train,
            xs_test=xs_test,
            ys_test=ys_test,
        )

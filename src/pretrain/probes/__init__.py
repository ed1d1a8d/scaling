"""Probes learn on top of learnt embeddings."""

import dataclasses
import pickle

import numpy as np


@dataclasses.dataclass
class EmbeddingDataset:
    xs_train: np.ndarray
    ys_train: np.ndarray
    xs_test: np.ndarray
    ys_test: np.ndarray

    dataset_id: str
    embedder_id: str
    n_embedder_params: int = 0

    def save_to_file(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dataclasses.asdict(self), f)

    @classmethod
    def load_from_file(cls, path: str):
        with open(path, "rb") as f:
            dict_ = pickle.load(f)
        return cls(**dict_)

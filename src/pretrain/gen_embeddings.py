"""Generic script to generate embeddings for a given dataset."""

import dataclasses
import os

import numpy as np
import torch
import torch.utils.data
import wandb
from simple_parsing import ArgumentParser, subgroups
from tqdm.auto import tqdm

from src.pretrain.datasets import BaseDatasetConfig, get_dataset_index
from src.pretrain.datasets.embedding import EmbeddingDataset
from src.pretrain.models import BaseEmbedderConfig, get_embedder_index
from src.pretrain.models.base import BaseEmbedder


@dataclasses.dataclass
class Config:
    """Configuration for generating embeddings."""

    embedder_cfg: BaseEmbedderConfig = subgroups(get_embedder_index())
    dataset_cfg: BaseDatasetConfig = subgroups(get_dataset_index())

    batch_size: int = 128
    num_workers: int = 12
    pin_memory: bool = False

    save_dir: str = "/home/gridsan/groups/ccg/data/scaling/embeddings"
    wandb_dir: str = "/home/gridsan/groups/ccg"
    tags: tuple[str, ...] = ("test",)

    @property
    def full_save_path(self) -> str:
        return os.path.join(
            self.save_dir,
            self.dataset_cfg.id,
            self.embedder_cfg.id.replace("/", "--") + ".pkl",
        )


def embed_dataset(
    embedder: BaseEmbedder,
    ds: torch.utils.data.Dataset,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    embeddings_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dl):
            xs_batch, ys_batch = cfg.dataset_cfg.parse_batch(batch)
            embeddings_batch = embedder.get_embeddings(xs_batch.cuda())

            embeddings_list.append(embeddings_batch.cpu().numpy())
            ys_list.append(ys_batch.cpu().numpy())

    embeddings = np.concatenate(embeddings_list)
    ys = np.concatenate(ys_list)

    return embeddings, ys


def main(cfg: Config):
    print(f"Loading embedder {cfg.embedder_cfg.id}...")
    embedder: BaseEmbedder = cfg.embedder_cfg.get_model()
    embedder.cuda().eval()

    print(f"Loading dataset {cfg.dataset_cfg.id}...")
    ds_train = cfg.dataset_cfg.get_train_ds(embedder.preprocess)
    ds_test = cfg.dataset_cfg.get_test_ds(embedder.preprocess)

    print("Embedding train set...")
    xs_train, ys_train = embed_dataset(embedder, ds_train, cfg)

    print("Embedding test set...")
    xs_test, ys_test = embed_dataset(embedder, ds_test, cfg)

    # Dictionary to save
    embedding_ds = EmbeddingDataset(
        xs_train=xs_train,
        ys_train=ys_train,
        xs_test=xs_test,
        ys_test=ys_test,
        dataset_id=cfg.dataset_cfg.id,
        embedder_id=cfg.embedder_cfg.id,
        n_embedder_params=embedder.n_embedder_params,
    )

    # Create cfg.save_dir / cfg.dataset_cfg.id
    os.makedirs(
        os.path.join(cfg.save_dir, cfg.dataset_cfg.id),
        exist_ok=True,
    )

    print("Saving embeddings...")
    embedding_ds.save_to_file(cfg.full_save_path)


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="gen-embeddings",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Run main logic
    main(cfg)

"""Generic script to generate embeddings for a given dataset."""

import dataclasses
import os

import numpy as np
import torch
import torch.utils.data
from simple_parsing import ArgumentParser, subgroups
from tqdm.auto import tqdm

from src.pretrain.datasets import BaseDatasetConfig
from src.pretrain.datasets.vision import cifar10, cifar100, imagenette, svhn
from src.pretrain.models import BaseEmbedder, BaseEmbedderConfig
from src.pretrain.models.vision import laion_clip, msft_beit, openai_clip
from src.pretrain.probes import EmbeddingDataset


@dataclasses.dataclass
class Config:
    """Configuration for generating embeddings."""

    # Embedder configuration
    embedder_cfg: BaseEmbedderConfig = subgroups(
        {
            "laion_clip": laion_clip.LaionClipConfig,
            "msft_beit": msft_beit.MsftBeitConfig,
            "openai_clip": openai_clip.OpenaiClipConfig,
        }
    )

    dataset_cfg: BaseDatasetConfig = subgroups(
        {
            "cifar10": cifar10.CIFAR10,
            "cifar100": cifar100.CIFAR100,
            "imagenette": imagenette.Imagenette,
            "svhn": svhn.SVHN,
        }
    )

    batch_size: int = 128
    num_workers: int = 12

    save_dir: str = "/home/gridsan/groups/ccg/data/scaling/embeddings"

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
        pin_memory=True,
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

    # Run main logic
    main(cfg)

"""Tests all datasets and models are compatible with each other."""


from typing import Any, Callable, Type

import torch
import torch.utils.data
import transformers

from src.pretrain.datasets import BaseDatasetConfig
from src.pretrain.datasets.vision import cifar10, cifar100, imagenette, svhn
from src.pretrain.models import BaseEmbedder, BaseEmbedderConfig
from src.pretrain.models.vision import laion_clip, msft_beit, openai_clip


def run_single_test(
    embedder: BaseEmbedder,
    ds: torch.utils.data.Dataset,
    batch_parser: Callable[[Any], tuple[torch.Tensor, torch.Tensor]],
):
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
    )

    i = 0
    for batch in dl:
        xs_batch, ys_batch = batch_parser(batch)
        assert len(xs_batch) == len(ys_batch)

        with torch.no_grad():
            embeddings_batch = embedder.get_embeddings(xs_batch.cuda())

        assert embeddings_batch.shape == (len(xs_batch), embedder.embed_dim)

        i += 1
        if i >= 3:
            break


def main():
    embedder_config_ts: list[Type[BaseEmbedderConfig]] = [
        msft_beit.MsftBeitConfig,
        laion_clip.LaionClipConfig,
        openai_clip.OpenaiClipConfig,
    ]

    ds_config_ts: list[Type[BaseDatasetConfig]] = [
        cifar10.CIFAR10,
        cifar100.CIFAR100,
        imagenette.Imagenette,
        svhn.SVHN,
    ]

    for ds_config_t in ds_config_ts:
        ds_config = ds_config_t()  # type: ignore

        for embedder_cfg_t in embedder_config_ts:
            for id in embedder_cfg_t().valid_model_ids:  # type: ignore
                embedder_cfg = embedder_cfg_t(id=id)

                print(f"Testing {ds_config.id} x {embedder_cfg.id}...")

                embedder = embedder_cfg.get_model().cuda().eval()
                assert embedder.n_embedder_params > 0

                ds_train = ds_config.get_train_ds(embedder.preprocess)
                ds_test = ds_config.get_test_ds(embedder.preprocess)

                run_single_test(embedder, ds_train, ds_config.parse_batch)
                run_single_test(embedder, ds_test, ds_config.parse_batch)

                print("OK")
                print()


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    main()

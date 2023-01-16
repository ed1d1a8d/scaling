"""Tests freezing of models."""
from typing import Type

import transformers

from src import utils
from src.pretrain.models import BaseEmbedderConfig, get_embedder_index


def main():
    embedder_config_ts: tuple[Type[BaseEmbedderConfig]] = tuple(
        get_embedder_index().values()
    )

    for embedder_cfg_t in embedder_config_ts:
        for id in embedder_cfg_t().valid_model_ids:  # type: ignore
            embedder_cfg = embedder_cfg_t(id=id)
            print(f"Testing {embedder_cfg.id}...")

            embedder = embedder_cfg.get_model().cuda().eval()
            assert embedder.n_embedder_params > 0

            try:
                assert (
                    embedder.n_embedder_params
                    == utils.count_params(embedder)
                    == sum(
                        utils.count_params(x)
                        for x in embedder.get_layers_for_freezing()
                    )
                )
                print("OK")
            except NotImplementedError:
                print("Not implemented. Skipping...")


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    main()

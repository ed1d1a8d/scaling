"""
Figures out which embeddings have not been generated and launches slurm jobs
to generate them.
"""

import dataclasses
import pathlib
from typing import Type

from simple_parsing import ArgumentParser

from src.pretrain import gen_embeddings
from src.pretrain.datasets import BaseDatasetConfig
from src.pretrain.models import BaseEmbedderConfig
from src.slurm import sbatch


@dataclasses.dataclass
class Config:
    n_nodes: int = 10
    max_concurrent: int = 1
    batch_size: int = 32
    log_dir: str = "gen_embeddings"

    dry_run: bool = False


def main(cfg: Config):

    embedder_cfg_t_dict: dict[
        str, Type[BaseEmbedderConfig]
    ] = gen_embeddings.Config.__dataclass_fields__["embedder_cfg"].metadata[
        "subgroups"
    ]

    dataset_cfg_t_dict: dict[
        str, Type[BaseDatasetConfig]
    ] = gen_embeddings.Config.__dataclass_fields__["dataset_cfg"].metadata[
        "subgroups"
    ]

    commands: list[str] = []
    for ds_key, dataset_cfg_t in dataset_cfg_t_dict.items():
        dataset_cfg = dataset_cfg_t()  # type: ignore

        for embedder_key, embedder_cfg_t in embedder_cfg_t_dict.items():
            for id in embedder_cfg_t().valid_model_ids:  # type: ignore
                embedder_cfg = embedder_cfg_t(id=id)

                # Check if the embeddings have already been generated.
                exp_config = gen_embeddings.Config(
                    dataset_cfg=dataset_cfg,
                    embedder_cfg=embedder_cfg,
                )
                if pathlib.Path(exp_config.full_save_path).exists():
                    print(
                        "Embeddings already exist:", exp_config.full_save_path
                    )
                    continue

                # If not, add a command to the list of commands to be run.
                command = " ".join(
                    [
                        "python -m src.pretrain.gen_embeddings",
                        f"--dataset_cfg {ds_key}",
                        f"--embedder_cfg {embedder_key}",
                        f"--embedder_cfg.id {id}",
                        f"--batch_size {cfg.batch_size}",
                    ]
                )
                commands.append(command)

    # Launch the commands.
    if cfg.dry_run:
        for command in commands:
            print(command)
    else:
        sbatch.launch_sharded_experiments(
            commands=commands,
            n_nodes=cfg.n_nodes,
            max_concurrent=cfg.max_concurrent,
            log_dir=cfg.log_dir,
        )


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    main(cfg)

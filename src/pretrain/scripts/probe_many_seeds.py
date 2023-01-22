"""Runs many seeds for a probe_embeddings experiment."""

import dataclasses
import pathlib

from simple_parsing import ArgumentParser, subgroups

from src.pretrain import gen_embeddings
from src.pretrain.datasets import (
    BaseDatasetConfig,
    get_dataset_index,
    get_dataset_key,
)
from src.pretrain.models import (
    BaseEmbedderConfig,
    get_embedder_index,
    get_embedder_key,
)
from src.slurm import sbatch


@dataclasses.dataclass
class Config:
    embedder_cfg: BaseEmbedderConfig = subgroups(get_embedder_index())
    dataset_cfg: BaseDatasetConfig = subgroups(get_dataset_index())
    n_classes: tuple[int, ...] = (10,)

    seed_start: int = 0
    seed_end: int = 50

    n_gpus: int = 1
    n_cpus: int = 20

    n_nodes: int = 11
    max_concurrent: int = 1
    log_dir: str = "probe_embeddings"

    dry_run: bool = False
    tags: tuple[str, ...] = ("probe-seeds-v1",)


def main(cfg: Config):
    # Check if the embeddings have already been generated.
    exp_config = gen_embeddings.Config(
        dataset_cfg=cfg.dataset_cfg,
        embedder_cfg=cfg.embedder_cfg,
    )
    if not pathlib.Path(exp_config.full_save_path).exists():
        print("Embeddings don't exist:", exp_config.full_save_path)
        raise ValueError

    commands: list[str] = []
    for seed in range(cfg.seed_start, cfg.seed_end):
        command = " ".join(
            [
                "python -m src.pretrain.probe_embeddings",
                f"--dataset_cfg {get_dataset_key(cfg.dataset_cfg)}",
                f"--embedder_cfg {get_embedder_key(cfg.embedder_cfg)}",
                f"--embedder_cfg.id {cfg.embedder_cfg.id}",
                f"--seed {seed}",
                f"--n_classes {' '.join(map(str, cfg.n_classes))}",
                f"--tags {' '.join(cfg.tags)}",
            ]
        )
        commands.append(command)

    # Launch commands
    sbatch.fancy_launch(
        commands=commands,
        n_nodes=cfg.n_nodes,
        max_concurrent=cfg.max_concurrent,
        log_dir=cfg.log_dir,
        n_gpus=cfg.n_gpus,
        n_cpus=cfg.n_cpus,
        dry_run=cfg.dry_run,
        interactive=False,
    )


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    main(cfg)

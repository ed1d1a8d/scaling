"""
Launches a sweep of finetuning experiments varying n_train.
"""

import dataclasses
from typing import Optional

from simple_parsing import ArgumentParser, subgroups

from src import utils
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
    # Set these to configure which embedder x dataset to use
    embedder_cfg: BaseEmbedderConfig = subgroups(get_embedder_index())
    dataset_cfg: BaseDatasetConfig = subgroups(get_dataset_index())

    # BEGIN specific finetune overrides
    init_with_trained_linear_probe: bool = True
    batch_size: int = 50
    seed: int = 0
    # END specific finetune overrides

    # Minimum n_train size for sweep
    n_train_start: int = 50

    # Possibly set n_val and add it to n_train
    additive_n_val: Optional[int] = None

    n_gpus: int = 1
    n_cpus: int = 20

    n_nodes: int = 11
    max_concurrent: int = 1
    log_dir: str = "finetune"

    dry_run: bool = False
    interactive: bool = False
    tags: tuple[str, ...] = ("finetune-sweep-v1",)

    def gen_n_trains(self):
        max_size = len(self.dataset_cfg.get_train_ds(lambda x: x))  # type: ignore

        base: int = self.n_train_start
        while True:
            for i in [1, 2, 5]:
                y = base * i + (
                    self.additive_n_val
                    if self.additive_n_val is not None
                    else 0
                )
                if y >= max_size:
                    yield max_size
                    return
                yield y
            base *= 10


def main(cfg: Config):

    commands: list[str] = []
    for n_train in cfg.gen_n_trains():
        command = " ".join(
            [
                "python -m src.pretrain.finetune",
                f"--dataset_cfg {get_dataset_key(cfg.dataset_cfg)}",
                f"--embedder_cfg {get_embedder_key(cfg.embedder_cfg)}",
                f"--embedder_cfg.id {cfg.embedder_cfg.id}",
                f"--init_with_trained_linear_probe {cfg.init_with_trained_linear_probe}",
                f"--batch_size {cfg.batch_size}",
                f"--seed {cfg.seed}",
                f"--n_train {n_train}",
                f"--tags {' '.join(cfg.tags)}",
                f"--n_val_override {cfg.additive_n_val}"
                if cfg.additive_n_val
                else "",
            ]
        )
        commands.append(command)

    if cfg.interactive:
        commands = [c for c in commands if utils.interactive_binary_query(c)]

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
            n_gpus=cfg.n_gpus,
            n_cpus=cfg.n_cpus,
        )


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    main(cfg)

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
    lr: float = 1e-5
    # END specific finetune overrides

    # Freezing configuration
    n_freezes: tuple[int, ...] = (18,)

    # fc_probe_cfg
    n_layerss: tuple[int, ...] = (1,)
    hidden_dims: tuple[int, ...] = (512,)

    # Minimum n_train size for sweep
    n_train_start: int = 50

    # Possibly set n_val and add it to n_train
    additive_n_val: Optional[int] = None

    n_gpus: int = 1
    n_cpus: int = 20

    max_nodes: int = 10
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
        for n_freeze in cfg.n_freezes:
            for n_layers in cfg.n_layerss:
                for hidden_dim in cfg.hidden_dims:
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
                            f"--n_layers {n_layers}",
                            f"--hidden_dim {hidden_dim}",
                            f"--lr {cfg.lr}",
                            f"--n_layers_to_freeze {n_freeze}",
                            f"--tags {' '.join(cfg.tags)}",
                            f"--n_val_override {cfg.additive_n_val}"
                            if cfg.additive_n_val
                            else "",
                        ]
                    )
                    commands.append(command)

    # Launch commands on slurm
    sbatch.fancy_launch(
        commands=commands,
        n_nodes=min(cfg.max_nodes, len(commands)),
        max_concurrent=cfg.max_concurrent,
        log_dir=cfg.log_dir,
        n_gpus=cfg.n_gpus,
        n_cpus=cfg.n_cpus,
        dry_run=cfg.dry_run,
        interactive=cfg.interactive,
    )


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    main(cfg)

"""
Launches many resnet_baseline runs.
Adapted from https://github.com/ed1d1a8d/miax/blob/main/miax/experiments/cfax_launcher.py.
"""

import dataclasses
import functools
import os
import shlex
import subprocess
from multiprocessing import current_process

import git.repo
from simple_parsing import ArgumentParser
from tqdm.contrib.concurrent import process_map

from src.pretrain.datasets import get_dataset_index

GIT_ROOT = git.repo.Repo(".", search_parent_directories=True).working_tree_dir


@dataclasses.dataclass
class Config:
    dataset_cfg: str = "cifar10"

    seed_start: int = 0  # inclusive
    seed_end: int = 1  # exclusive

    n_train_min: int = 384
    n_train_max: int = 50_000
    n_val: int = 256

    weight_decay: float = 1e-4

    tags: tuple[str, ...] = ("try-1",)

    max_workers: int = 30
    gpu_start_idx: int = 0  # inclusive
    gpu_end_idx: int = 6  # exclusive

    dry_run: bool = False

    def __post_init__(self):
        assert self.dataset_cfg in get_dataset_index()
        assert self.seed_end > self.seed_start
        assert self.gpu_end_idx > self.gpu_start_idx

    def gen_n_trains(self):
        base = 1
        while True:
            for i in range(1, 10):
                if base * i < self.n_train_min:
                    continue
                elif base * i >= self.n_train_max:
                    yield self.n_train_max
                    return
                else:
                    yield base * i
            base *= 10


def run_command(command: str, gpu_start_idx: int, gpu_end_idx: int):
    # See https://stackoverflow.com/a/48389099
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(
        gpu_start_idx
        + current_process()._identity[0] % (gpu_end_idx - gpu_start_idx)
    )

    subprocess.run(
        shlex.split(command),
        cwd=GIT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def main(cfg: Config):
    commands: list[str] = []
    for seed in range(cfg.seed_start, cfg.seed_end):
        for n_train in cfg.gen_n_trains():
            command = " ".join(
                [
                    "python -m src.pretrain.resnet_baseline",
                    f"--dataset_cfg {cfg.dataset_cfg}",
                    f"--n_train {n_train}",
                    f"--n_val_override {cfg.n_val}",
                    f"--weight_decay {cfg.weight_decay}",
                    f"--seed {seed}",
                    f"--tags {' '.join(cfg.tags)}",
                ]
            )
            commands.append(command)

    if cfg.dry_run:
        for command in commands:
            print(command)
        print(f"Attempting to launch {len(commands)} commands...")
    else:
        print(f"Attempting to launch {len(commands)} commands...")
        process_map(
            functools.partial(
                run_command,
                gpu_start_idx=cfg.gpu_start_idx,
                gpu_end_idx=cfg.gpu_end_idx,
            ),
            commands,
            max_workers=cfg.max_workers,
        )


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    main(cfg)

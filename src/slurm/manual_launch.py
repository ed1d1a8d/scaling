"""
Launches a sweep of finetuning experiments varying n_train.
"""

import dataclasses
import sys

from simple_parsing import ArgumentParser

from src.slurm import sbatch


@dataclasses.dataclass
class Config:
    n_gpus: int = 1
    n_cpus: int = 20

    max_nodes: int = 10
    max_concurrent: int = 1
    log_dir: str = "manual"

    dry_run: bool = False


def main(cfg: Config):
    # Read commands from stdin
    commands: list[str] = [c.strip() for c in sys.stdin.readlines()]

    # Launch the commands.
    if cfg.dry_run:
        for command in commands:
            print(command)
        print(f"Would have launched {len(commands)} commands.")
    else:
        sbatch.launch_sharded_experiments(
            commands=commands,
            n_nodes=min(cfg.max_nodes, len(commands)),
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

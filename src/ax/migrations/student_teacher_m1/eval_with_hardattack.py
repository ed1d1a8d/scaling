"""
This script goes through student-teacher-v1 experiments and evaluates runs
that use soft-labels with a hard label attack as well.
"""
import dataclasses
import os
from typing import NamedTuple

import torch
import wandb
import wandb.apis
from simple_parsing import ArgumentParser
from src.ax.attack.FastPGD import FastPGD
from src.ax.attack.teacher_loss import TeacherCrossEntropy
from src.ax.train import ExperimentConfig, evaluate, tag_dict
from src.utils import runs_to_df, wandb_run_save_objs
from torch import nn

MIGRATION_TAG = "student_teacher_migration1"

API = wandb.Api()
WANDB_DIR = "/home/gridsan/groups/ccg/wandb"
ID_TO_CKPT = {
    f.split("-")[-1]: os.path.join(WANDB_DIR, f, "files/model.ckpt")
    for f in os.listdir(WANDB_DIR)
    if os.path.isdir(os.path.join(WANDB_DIR, f))
}


@dataclasses.dataclass
class Config:
    row_mod: int = 0
    num_jobs: int = 0
    seed: int = 42


class _ExperimentRow(NamedTuple):
    """Just for typing purposes"""

    run_path: str
    id: str


def update_run(row: _ExperimentRow):
    print("Updating:", row.run_path)
    ckpt_file = ID_TO_CKPT[row.id]
    assert os.path.exists(ckpt_file)

    run: wandb.apis.public.Run = API.run(row.run_path)
    if run.config.get(MIGRATION_TAG, False):
        print(f"Skipping {row.id} because already migrated.")
        return

    torch.cuda.empty_cache()

    cfg = ExperimentConfig(
        teacher_ckpt_path="/home/gridsan/groups/ccg/wandb/run-reconstructed-1s50k73h/files/model.ckpt",
        teacher_use_softmax=True,
    )

    try:
        net: nn.Module = cfg.get_net().cuda()
        net = net.to(memory_format=torch.channels_last)  # type: ignore
        net.load_state_dict(torch.load(ckpt_file))
        net.eval()
    except RuntimeError:
        print(f"Failed to load weights for {row.id}. Corrupted file!")
        return

    teacher_net: nn.Module = cfg.get_net().cuda()
    teacher_net = teacher_net.to(memory_format=torch.channels_last)  # type: ignore
    teacher_net.load_state_dict(torch.load(cfg.teacher_ckpt_path))
    teacher_net.eval()

    attack_test = FastPGD(
        model=net,
        eps=cfg.adv_eps_eval,
        alpha=cfg.adv_eps_eval / cfg.pgd_steps * 2.3
        if cfg.pgd_steps > 0
        else 0,
        steps=cfg.pgd_steps,
        random_start=True,
        loss=TeacherCrossEntropy(teacher_net, use_softmax=False),
    )

    test_loaders = cfg.get_test_loaders()
    for split_name, loader in test_loaders.items():
        print(f"Starting evaluation of {split_name} split...")
        test_dict, test_imgs = evaluate(net, loader, attack_test, cfg, None)

        test_metrics = {
            k: v.data
            for k, v in tag_dict(
                test_dict, prefix=f"{split_name}_", suffix=f"_hardattack"
            ).items()
        }

        # Update run
        run.summary.update(test_metrics)
        wandb_run_save_objs(
            run,
            {
                f"media/api/{split_name}_imgs_hardattack/{i}.png": img.image
                for i, img in enumerate(test_imgs)
            }
            | {
                f"media/api/{split_name}_imgs_hardattack/{i}_caption.txt": img._caption
                for i, img in enumerate(test_imgs)
            },  # type: ignore
        )
        run.update()

        print(f"Finished evaluation of {split_name} split.")

    run.config.update({MIGRATION_TAG: True})
    run.update()


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    # Collect runs
    runs: list[wandb.apis.public.Run] = API.runs(
        f"data-frugal-learning/adv-train",
        filters={
            "$and": [
                {"tags": {"$in": ["student-teacher-v1"]}},
                {"state": {"$in": ["finished"]}},
                {"config.teacher_use_softmax": {"$eq": True}},
            ]
        },
        # filters={"tags": {"$in": ["test"]}},
    )
    print("Total runs:", len(runs))

    # Parse runs into dataframe
    df = runs_to_df(runs).sort_values("id").reset_index(drop=True)

    all_rows: list[_ExperimentRow] = list(df.itertuples())
    assigned_rows = all_rows[cfg.row_mod :: cfg.num_jobs]

    # Update assigned runs
    print("Assigned runs:", [row.id for row in assigned_rows])
    for row in assigned_rows:
        update_run(row)


if __name__ == "__main__":
    main()

"""
This script goes through all old wandb runs and recomputes test statistics with
autoattack.
"""
import dataclasses
import os
from typing import NamedTuple

import torch
import wandb
import wandb.apis
from simple_parsing import ArgumentParser
from src.ax.attack.FastAutoAttack import FastAutoAttack
from src.ax.train import DatasetT, ExperimentConfig, ModelT, evaluate, tag_dict
from src.utils import runs_to_df, wandb_run_save_objs
from torch import nn

API = wandb.Api()
WANDB_DIR = "/home/gridsan/groups/ccg/wandb"
ID_TO_CKPT = {
    f.split("-")[-1]: os.path.join(WANDB_DIR, f, "files/model.ckpt")
    for f in os.listdir(WANDB_DIR)
    if os.path.isdir(os.path.join(WANDB_DIR, f))
}


@dataclasses.dataclass
class Config:
    row_start: int = 0
    row_end: int = 0  # Non-inclusive
    seed: int = 42


class _ExperimentRow(NamedTuple):
    """Just for typing purposes"""

    run_path: str
    id: str
    dataset: str
    model: str
    depth: int
    width: int
    adv_eps_eval: float


def update_run(row: _ExperimentRow):
    print("Updating:", row.run_path)
    ckpt_file = ID_TO_CKPT[row.id]
    assert os.path.exists(ckpt_file)

    run: wandb.apis.public.Run = API.run(row.run_path)
    torch.cuda.empty_cache()

    dataset_t = DatasetT[row.dataset.split(".")[-1]]
    model_t = ModelT[row.model.split(".")[-1]]

    cfg = ExperimentConfig(
        dataset=dataset_t,
        model=model_t,
        depth=row.depth,
        width=row.width,
        adv_eps_eval=row.adv_eps_eval,
        eval_batch_size=512,  # To avoid OOM with AutoAttack,
    )

    net: nn.Module = cfg.get_net().cuda()
    net = net.to(memory_format=torch.channels_last)  # type: ignore
    net.load_state_dict(torch.load(ckpt_file))
    net.eval()

    attack_test = FastAutoAttack(net, eps=cfg.adv_eps_eval, seed=cfg.seed)

    test_loaders = cfg.get_test_loaders()
    for split_name, loader in test_loaders.items():
        print(f"Starting evaluation of {split_name} split...")
        test_dict, test_imgs = evaluate(net, loader, attack_test, cfg)

        test_metrics = {
            k: v.data
            for k, v in tag_dict(test_dict, prefix=f"{split_name}_").items()
        }
        old_test_metrics = tag_dict(
            {k: run.summary[k] for k in test_metrics.keys()}, suffix="_pgd_old"
        )

        wandb.Image
        # Update run
        run.summary.update(test_metrics | old_test_metrics)
        wandb_run_save_objs(
            run,
            {
                f"media/api/{split_name}_imgs_autoattack/{i}.png": img.image
                for i, img in enumerate(test_imgs)
            }
            | {
                f"media/api/{split_name}_imgs_autoattack/{i}_caption.txt": img._caption
                for i, img in enumerate(test_imgs)
            },  # type: ignore
        )
        run.update()

        print(f"Finished evaluation of {split_name} split.")

    run.config.update({"use_autoattack": True, "retro_use_autoattack": True})
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
        # filters={
        #     "$and": [
        #         {"$not": {"tags": {"$in": ["test"]}}},
        #         {
        #             "config.dataset": {
        #                 "$in": ["DatasetT.CIFAR10", "DatasetT.CIFAR5m", None]
        #             }
        #         },
        #     ]
        # },
        filters={"tags": {"$in": ["test"]}},
    )

    # Parse runs into dataframe
    df = runs_to_df(runs).sort_values("id").reset_index(drop=True)
    df.loc[df.dataset.isna(), "dataset"] = "DatasetT.CIFAR5m"
    df.loc[df.model.isna(), "model"] = "ModelT.WideResNet"
    df.loc[df.adv_eps_eval.isna(), "adv_eps_eval"] = 8 / 255

    all_rows: list[_ExperimentRow] = list(df.itertuples())
    assigned_rows = all_rows[cfg.row_start : cfg.row_end]

    # Update assigned runs
    print("Assigned runs:", [row.id for row in assigned_rows])
    for row in assigned_rows:
        update_run(row)


if __name__ == "__main__":
    main()

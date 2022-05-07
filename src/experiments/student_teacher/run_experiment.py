import dataclasses
import logging
import math
import os
import warnings

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simple_parsing import ArgumentParser
from src.experiments.harmonics.data import (
    HypercubeDataModule,
    HypercubeDataModuleConfig,
)
from src.experiments.harmonics.fc_net import FCNet, FCNetConfig, HFReg
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclasses.dataclass
class ExperimentConfig:
    input_dim: int = 8

    teacher_layer_widths: tuple[int, ...] = (96, 192, 1)
    student_width_scale: float = 1.0
    teacher_seed: int = 100
    student_seed: int = 101

    teacher_sparsity: float = 0.05
    teacher_output_sparsity: float = 1

    # input_dim and layer_widths are overriden for net_cfg
    base_net_cfg: FCNetConfig = dataclasses.field(
        default_factory=lambda: FCNetConfig(
            high_freq_reg=HFReg.NONE,
            sched_monitor="train_mse",
            sched_verbose=True,
            l1_reg=True,
            l1_reg_lambda=3e-7,
            l2_reg=True,
            l2_reg_lambda=0,
        )
    )

    data_cfg: HypercubeDataModuleConfig = dataclasses.field(
        default_factory=lambda: HypercubeDataModuleConfig(
            input_dim=-1,
            n_train=100,
            n_val=1024,
            num_workers=0,
            batch_size=256,
            cube_lo=-1,
            cube_hi=1,
        )
    )

    training_seed: int = 42
    patience_steps: int = 1000

    viz_samples: int = 512  # Number of side samples for visualizations
    viz_pad: tuple[int, int] = (3, 3)  # Visualization padding
    viz_value: float = 0.42  # Vizualization value

    tags: tuple[str, ...] = ("test",)

    def __post_init__(self):
        assert sum(self.viz_pad) + 2 == self.input_dim

    @property
    def net_cfg(self) -> FCNetConfig:
        patience_ub_in_epochs = math.ceil(
            self.patience_steps / (self.data_cfg.n_train / self.data_cfg.batch_size)
        )
        return dataclasses.replace(
            self.base_net_cfg,
            input_dim=self.input_dim,
            sched_patience=patience_ub_in_epochs,
        )

    def get_teacher_net(self) -> FCNet:
        pl.seed_everything(self.teacher_seed)
        return FCNet(
            dataclasses.replace(
                self.net_cfg,
                layer_widths=self.teacher_layer_widths,
                sparsity=self.teacher_sparsity,
                output_sparsity=self.teacher_output_sparsity,
            )
        )

    def get_student_net(self) -> FCNet:
        pl.seed_everything(self.student_seed)
        return FCNet(
            dataclasses.replace(
                self.net_cfg,
                layer_widths=(
                    *(
                        int(self.student_width_scale * w)
                        for w in self.teacher_layer_widths[:-1]
                    ),
                    1,
                ),
            )
        )

    def get_dm(self, net: FCNet) -> HypercubeDataModule:
        return HypercubeDataModule(
            fn=net,
            cfg=dataclasses.replace(
                self.data_cfg,
                input_dim=self.input_dim,
            ),
        )


def viz_to_wandb(cfg: ExperimentConfig, net: FCNet, viz_name: str):
    net.viz_2d(
        side_samples=cfg.viz_samples,
        pad=cfg.viz_pad,
        value=cfg.viz_value,
        lo=cfg.data_cfg.cube_lo,
        hi=cfg.data_cfg.cube_hi,
    )
    wandb.log({viz_name: plt.gcf()})
    plt.clf()


class CustomLogger(pl.Callback):
    def __init__(self, log_every_n_steps: int):
        super().__init__()
        self.pbar = tqdm()

        self.log_every_n_batches = log_every_n_steps
        self.n_batches_seen = 0

    def on_train_batch_end(self, trainer: pl.Trainer, *_, **__):
        self.n_batches_seen += 1
        if self.n_batches_seen % self.log_every_n_batches == 0:
            md = {k: v.item() for k, v in trainer.logged_metrics.items()} | {
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
                "n_batches_seen": self.n_batches_seen,
            }
            wandb.log(md)

            self.pbar.update(self.log_every_n_batches)
            self.pbar.set_description(
                f"tr.loss={md['train_loss']: .6e}; tr.mse={md['train_mse']: .6e}"
            )


def train(dm: HypercubeDataModule, net: FCNet):
    trainer = pl.Trainer(
        gpus=1,
        deterministic=True,
        logger=False,  # We do custom logging instead.
        max_epochs=-1,
        callbacks=[
            CustomLogger(log_every_n_steps=10),
            EarlyStopping(
                monitor="train_loss",
                patience=max(
                    net.cfg.sched_patience + 10, int(1.5 * net.cfg.sched_patience)
                ),
                mode="min",
            ),
        ],
        enable_progress_bar=False,
        weights_summary=None,
    )

    trainer.fit(
        model=net,
        datamodule=dm,
    )


def evaluate(dm: HypercubeDataModule, net: FCNet):
    def _get_mse(dl: DataLoader):
        return pl.Trainer(
            gpus=1,
            logger=False,
            enable_progress_bar=False,
            weights_summary=None,
        ).test(model=net, dataloaders=dl, verbose=False,)[0]["test_mse"]

    wandb.run.summary["final_train_mse"] = _get_mse(dm.train_dataloader(shuffle=False))
    wandb.run.summary["final_val_mse"] = _get_mse(dm.val_dataloader())


def run_experiment(cfg: ExperimentConfig):
    teacher_net = cfg.get_teacher_net()
    torch.save(teacher_net.state_dict(), os.path.join(wandb.run.dir, "teacher.pt"))
    viz_to_wandb(cfg=cfg, net=teacher_net, viz_name="teacher")
    with torch.no_grad():
        wandb.run.summary["teacher_l0_norm"] = teacher_net.get_l0_norm().item()
        wandb.run.summary["teacher_l1_norm"] = teacher_net.get_l1_norm().item()
        wandb.run.summary["teacher_l2_norm2"] = teacher_net.get_l2_norm2().item()

    dm = cfg.get_dm(net=teacher_net)
    dm.setup()

    student_net = cfg.get_student_net()
    student_net.cfg.l1_reg_lim = teacher_net.get_l1_norm().item()
    student_net.cfg.l2_reg_lim = teacher_net.get_l2_norm2().item()
    viz_to_wandb(cfg=cfg, net=student_net, viz_name="student_init")

    pl.seed_everything(seed=cfg.training_seed, workers=True)
    train(dm, student_net)

    torch.save(student_net.state_dict(), os.path.join(wandb.run.dir, "student.pt"))
    viz_to_wandb(cfg=cfg, net=student_net, viz_name="student_trained")
    evaluate(dm, student_net)


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="student-teacher",
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Save all ckpt files immediately
    wandb.save("*.ckpt")

    # Run experiment
    run_experiment(cfg)


if __name__ == "__main__":
    main()

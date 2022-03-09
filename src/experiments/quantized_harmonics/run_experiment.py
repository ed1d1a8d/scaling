import dataclasses
import logging
import tempfile
import warnings
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simple_parsing import ArgumentParser
from src.experiments.quantized_harmonics.fc_net import FCNet, FCNetConfig
from src.experiments.quantized_harmonics.harmonics import (HarmonicFn,
                                                           HarmonicFnConfig)
from src.experiments.quantized_harmonics.qdata import QDataModule
from torch.utils.data.dataloader import DataLoader

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 2
    input_resolution: int = 96

    freq_limit: int = 3  # For ground truth harmonic function
    num_components: int = 16  # For ground truth harmonic function
    true_hf_seed: int = 42

    # (96, 192, 1) is from https://arxiv.org/pdf/2102.06701.pdf.
    layer_widths: tuple[int, ...] = (128, 128, 128, 1)

    train_sizes: tuple[int, ...] = (1, 10, 100, 1000, 8000)
    train_seed: int = 0

    learning_rate: float = 3e-3
    max_epochs: int = 9001
    batch_size: int = 256

    sched_patience: int = 50
    sched_decay: float = 0.1
    sched_min_lr: float = 1e-6

    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 100
    early_stopping_monitor: str = "val_mse"

    num_workers: int = 0

    viz_samples: int = 512  # Number of side samples for visualizations
    viz_pad: tuple[int, int] = (0, 0)  # Visualization padding
    viz_value: float = 0.42  # Vizualization value


REGISTERED_METRICS = set()


def wandb_custom_log(
    metrics: dict[str, float],
    step: int,
    step_name: str,
    metric_prefix: Optional[str] = None,
    metric_suffix: Optional[str] = None,
):
    global REGISTERED_METRICS

    if metric_prefix is not None:
        metrics = {f"{metric_prefix}/{k}": v for k, v in metrics.items()}
    if metric_suffix is not None:
        metrics = {f"{k}/{metric_suffix}": v for k, v in metrics.items()}

    if step_name not in REGISTERED_METRICS:
        wandb.define_metric(name=step_name, hidden=True)
        REGISTERED_METRICS.add(step_name)
    for metric_name in metrics.keys():
        if metric_name not in REGISTERED_METRICS:
            wandb.define_metric(name=metric_name, step_metric=step_name)
            REGISTERED_METRICS.add(metric_name)

    wandb.log(metrics | {step_name: step})


def aggregate_metrics(
    list_of_metrics: list[dict[str, float]],
    agg_fn: Callable[[list[float]], float],
) -> dict[str, float]:
    if len(list_of_metrics) == 0:
        return dict()
    return {
        k: agg_fn([m[k] for m in list_of_metrics]) for k in list_of_metrics[0].keys()
    }


class CustomLoggingCallback(pl.Callback):
    def __init__(
        self,
        tag_prefix: str,
        tag_suffix: str,
        n_train: int,
    ):
        self.tag_prefix = tag_prefix
        self.tag_suffix = tag_suffix
        self.n_train = n_train
        self.train_epochs_completed = 0

        self.best_val_mse = np.inf
        self.best_epochs: int = -1
        self.best_num_dps: int = -1

    def on_train_epoch_end(self, trainer: pl.Trainer, *_, **__):
        self.train_epochs_completed += 1

        wandb_custom_log(
            metrics=dict(
                train_mse=float(trainer.logged_metrics["train_mse"]),
                val_mse=float(trainer.logged_metrics["val_mse"]),
            ),
            step_name="n_dps_seen",
            step=self.n_train * self.train_epochs_completed,
            metric_prefix=self.tag_prefix,
            metric_suffix=self.tag_suffix,
        )

        val_mse = float(trainer.logged_metrics["val_mse"])
        if val_mse < self.best_val_mse:
            self.best_val_mse = val_mse
            self.best_epochs = self.train_epochs_completed
            self.best_num_dps = self.n_train * self.train_epochs_completed


@dataclasses.dataclass(frozen=True)
class TrainResult:
    # Final loss values
    train_mse: float
    val_mse: float
    epochs: int
    num_dps: int


def train_student(
    hf: HarmonicFn,
    n_train: int,
    cfg: ExperimentConfig,
    tag_prefix: str,
    tag_suffix: str,
) -> tuple[FCNet, TrainResult]:
    pl.seed_everything(cfg.train_seed, workers=True)

    student = FCNet(
        FCNetConfig(
            input_dim=cfg.input_dim,
            input_resolution=cfg.input_resolution,
            layer_widths=cfg.layer_widths,
            learning_rate=cfg.learning_rate,
            sched_monitor=cfg.early_stopping_monitor,
            sched_patience=cfg.sched_patience,
            sched_decay=cfg.sched_decay,
            sched_min_lr=cfg.sched_min_lr,
        )
    )
    dm = QDataModule(
        fn=hf,
        input_dim=hf.cfg.input_dim,
        input_resolution=cfg.input_resolution,
        n_train=n_train,
        seed=cfg.train_seed,
        num_workers=cfg.num_workers,
    )

    cc = CustomLoggingCallback(
        tag_prefix=tag_prefix,
        tag_suffix=tag_suffix,
        n_train=n_train,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        ckpt = ModelCheckpoint(monitor="val_mse", dirpath=tmpdirname)
        trainer = pl.Trainer(
            gpus=1,
            deterministic=True,
            logger=False,  # We do custom logging instead.
            log_every_n_steps=1,
            max_epochs=cfg.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor=cfg.early_stopping_monitor,
                    patience=cfg.early_stopping_patience,
                    mode="min",
                ),
                ckpt,
                cc,
            ],
            enable_progress_bar=False,
            weights_summary=None,
        )
        trainer.fit(
            model=student,
            datamodule=dm,
        )

        student = FCNet.load_from_checkpoint(ckpt.best_model_path)

    def _get_mse(dl: DataLoader):
        (test_dict,) = trainer.test(
            model=student,
            dataloaders=dl,
            verbose=False,
        )
        return test_dict["test_mse"]

    return student, TrainResult(
        train_mse=_get_mse(dm.train_dataloader(shuffle=False)),
        val_mse=_get_mse(dm.val_dataloader()),
        epochs=cc.best_epochs,
        num_dps=cc.best_num_dps,
    )


def run_experiment(cfg: ExperimentConfig):
    hf = HarmonicFn(
        cfg=HarmonicFnConfig(
            input_dim=cfg.input_dim,
            freq_limit=cfg.freq_limit,
            num_components=cfg.num_components,
            seed=cfg.true_hf_seed,
        )
    )

    # Save hf info
    hf.viz_2d(side_samples=cfg.viz_samples, pad=cfg.viz_pad, value=cfg.viz_value)
    wandb.log({"hf_viz": plt.gcf()})
    plt.cla()

    for n_train in cfg.train_sizes:
        student, tr = train_student(
            hf=hf,
            n_train=n_train,
            cfg=cfg,
            tag_prefix=f"train_details/n_{n_train:06}",
            tag_suffix="",
        )

        wandb_custom_log(
            dict(
                train_mse=tr.train_mse,
                val_mse=tr.val_mse,
                epochs_taken=tr.epochs,
                dps_taken=tr.num_dps,
            ),
            step=n_train,
            step_name="n_train_dps",
        )


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="quantized-harmonics",
        tags=["nn"],
        config=dataclasses.asdict(cfg),
    )

    # Run experiment
    run_experiment(cfg)


if __name__ == "__main__":
    main()

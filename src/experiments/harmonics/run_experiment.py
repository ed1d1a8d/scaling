import dataclasses
import logging
import os
import pickle
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
from src.experiments.harmonics.data import HypercubeDataModule
from src.experiments.harmonics.fc_net import FCNet, FCNetConfig, HFReg
from src.experiments.harmonics.harmonics import HarmonicFn, HarmonicFnConfig
from torch.utils.data.dataloader import DataLoader

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 2

    freq_limit: int = 3  # For ground truth harmonic function
    num_components: int = 16  # For ground truth harmonic function
    true_hf_seed: int = 42

    # (96, 192, 1) is from https://arxiv.org/pdf/2102.06701.pdf.
    layer_widths: tuple[int, ...] = (128, 128, 128, 1)

    n_val: int = 1024
    val_seed: int = -2

    train_sizes: tuple[int, ...] = tuple(int(2 ** x) for x in range(18))
    trials_per_size: int = 3

    learning_rate: float = 3e-3
    max_epochs: int = 9001
    batch_size: int = 256

    high_freq_reg: HFReg = HFReg.MCLS
    high_freq_lambda: float = 1
    high_freq_freq_limit: int = 2
    high_freq_mcls_samples: int = 1024
    high_freq_dft_ss: int = 8

    # TODO: Implement l2_reg
    # l2_reg: float = 0

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


def perfect_pow_2(n: int) -> bool:
    """Returns whether a number is an integer and is a perfect power of 2."""
    if not isinstance(n, int):
        return False

    if n < 1:
        return False

    while n > 1:
        if n % 2 != 0:
            return False
        n //= 2

    return True


def exact_log_2(n: int) -> int:
    assert perfect_pow_2(n)

    res = 0
    while n > 1:
        res += 1
        n //= 2

    return res


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
                train_hfn=float(trainer.logged_metrics["train_hfn"]),
                train_loss=float(trainer.logged_metrics["train_loss"]),
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
    seed: int,
) -> tuple[FCNet, TrainResult]:
    pl.seed_everything(seed, workers=True)

    student = FCNet(
        FCNetConfig(
            input_dim=cfg.input_dim,
            layer_widths=cfg.layer_widths,
            learning_rate=cfg.learning_rate,
            high_freq_reg=cfg.high_freq_reg,
            high_freq_lambda=cfg.high_freq_lambda,
            high_freq_freq_limit=cfg.high_freq_freq_limit,
            high_freq_mcls_samples=cfg.high_freq_mcls_samples,
            high_freq_dft_ss=cfg.high_freq_dft_ss,
            sched_monitor=cfg.early_stopping_monitor,
            sched_patience=cfg.sched_patience,
            sched_decay=cfg.sched_decay,
            sched_min_lr=cfg.sched_min_lr,
        )
    )
    dm = HypercubeDataModule(
        fn=hf,
        input_dim=hf.cfg.input_dim,
        n_train=n_train,
        n_val=cfg.n_val,
        train_seed=n_train,
        val_seed=cfg.val_seed,
        num_workers=cfg.num_workers,
    )

    cc = CustomLoggingCallback(tag_prefix=tag_prefix, tag_suffix=tag_suffix, n_train=n_train,)

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
    assert all(perfect_pow_2(n) for n in cfg.train_sizes)

    # Dump config, makes it easier to load
    with open(os.path.join(wandb.run.dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)
    wandb.save(os.path.join(wandb.run.dir, "config.pkl"), policy="now")

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
        metrics_by_trial: list[dict[str, float]] = []
        for trial in range(cfg.trials_per_size):
            print(f"Training for {n_train=}, {trial=}")

            n_train_tag = f"n_{n_train:07}"
            trial_tag = f"trial_{trial:02}"

            student, tr = train_student(
                hf=hf,
                n_train=n_train,
                cfg=cfg,
                tag_prefix=f"train_details/{n_train_tag}",
                tag_suffix=trial_tag,
                seed=trial,
            )

            print(tr)
            metrics = dict(
                train_mse=tr.train_mse,
                val_mse=tr.val_mse,
                epochs=tr.epochs,
                num_dps=tr.num_dps,
            )
            metrics_by_trial.append(metrics)
            wandb_custom_log(
                metrics=metrics,
                step=n_train,
                step_name="n_train_dps",
                metric_prefix="summary",
                metric_suffix=f"{trial:02}",
            )

            student.viz_2d(
                side_samples=cfg.viz_samples,
                pad=cfg.viz_pad,
                value=cfg.viz_value,
            )
            wandb.log({f"student_viz/{n_train_tag}/{trial_tag}": plt.gcf()})
            plt.cla()

        wandb_custom_log(
            metrics=aggregate_metrics(metrics_by_trial, agg_fn=np.median),
            step=n_train,
            step_name="n_train_dps",
            metric_prefix="summary",
            metric_suffix="median",
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
        project="harmonic-learning-2d",
        tags=["nn", "regularization"],
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Run experiment
    run_experiment(cfg)


if __name__ == "__main__":
    main()

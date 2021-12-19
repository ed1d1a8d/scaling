import dataclasses
import logging
import pathlib
import pickle
import tempfile
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import src.utils as utils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simple_parsing import ArgumentParser
from src.experiments.harmonics.fc_net import FCNet, FCNetConfig
from src.experiments.harmonics.harmonics import (
    HarmonicDataModule,
    HarmonicFn,
    HarmonicFnConfig,
)
from torch.utils.data.dataloader import DataLoader

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 4

    freq_limit: int = 2  # For ground truth harmonic function
    num_components: int = 4  # For ground truth harmonic function
    true_hf_seed: int = -1

    # (96, 192, 1) is from https://arxiv.org/pdf/2102.06701.pdf.
    layer_widths: tuple[int, ...] = (128, 128, 128, 1)  # 1024, 42, 5, 1)

    n_val: int = 1024
    val_seed: int = -2

    train_sizes: tuple[int, ...] = tuple(int(2 ** x) for x in range(18))
    trials_per_size: int = 3

    learning_rate: float = 3e-3
    max_epochs: int = 9001
    batch_size: int = 256
    l2_reg: float = 0

    sched_patience: int = 50
    sched_decay: float = 0.1
    sched_min_lr: float = 1e-6

    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 100
    early_stopping_monitor: str = "val_mse"

    num_workers: int = 0

    viz_samples: int = 512  # Number of side samples for visualizations
    viz_pad: tuple[int, int] = (1, 1)  # Visualization padding
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


class CustomMLFlowCallback(pl.Callback):
    def __init__(
        self,
        tag: str,
        n_train: int,
    ):
        self.tag = tag
        self.n_train = n_train
        self.train_epochs_completed = 0

        self.best_val_mse = np.inf
        self.best_epochs: Optional[int] = None
        self.best_num_dps: Optional[int] = None

    def tag_dict(self, **kwargs) -> dict[str, float]:
        # The zzz at the front is to make it go to the end of the mlflow logs.
        return {f"zzz_{self.tag}_{k}": v for k, v in kwargs.items()}

    def on_train_epoch_end(self, trainer: pl.Trainer, *_, **__):
        self.train_epochs_completed += 1

        mlflow.log_metrics(
            metrics=self.tag_dict(
                train_mse=float(trainer.logged_metrics["train_mse"]),
                val_mse=float(trainer.logged_metrics["val_mse"]),
            ),
            step=self.n_train * self.train_epochs_completed,
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
    tag: str,
    seed: int,
) -> tuple[FCNet, TrainResult]:
    pl.seed_everything(seed, workers=True)

    student = FCNet(
        FCNetConfig(
            input_dim=cfg.input_dim,
            layer_widths=cfg.layer_widths,
            learning_rate=cfg.learning_rate,
            sched_monitor=cfg.early_stopping_monitor,
            sched_patience=cfg.sched_patience,
            sched_decay=cfg.sched_decay,
            sched_min_lr=cfg.sched_min_lr,
        )
    )
    dm = HarmonicDataModule(
        hf=hf,
        n_train=n_train,
        n_val=cfg.n_val,
        train_seed=n_train,
        val_seed=cfg.val_seed,
        num_workers=cfg.num_workers,
    )

    cc = CustomMLFlowCallback(tag=tag, n_train=n_train)

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

    # Dump config
    with open(mlflow.get_artifact_uri("config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

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
    mlflow.log_figure(figure=plt.gcf(), artifact_file="hf-viz.png")
    plt.cla()
    with open(mlflow.get_artifact_uri("hf.pkl"), "wb") as f:
        pickle.dump(hf, f)

    for n_train in cfg.train_sizes:
        for trial in range(cfg.trials_per_size):
            print(f"Training for {n_train=}, {trial=}")
            # n_tag = f"n_{n_train:07}"
            full_tag = f"n_{n_train:07}_trial_{trial:02}"

            student, tr = train_student(
                hf=hf,
                n_train=n_train,
                cfg=cfg,
                tag=full_tag,
                seed=trial,
            )

            def _add_log10s(d: dict[str, float]) -> dict[str, float]:
                return d | {f"log10_{k}": np.log10(v + 1e-42) for k, v in d.items()}

            print(tr)
            mlflow.log_metrics(
                _add_log10s(
                    dict(
                        train_mse=tr.train_mse,
                        val_mse=tr.val_mse,
                        epochs=tr.epochs,
                        num_dps=tr.num_dps,
                    )
                ),
                step=exact_log_2(n_train),
            )

            student.viz_2d(
                side_samples=cfg.viz_samples,
                pad=cfg.viz_pad,
                value=cfg.viz_value,
            )
            mlflow.log_figure(
                figure=plt.gcf(),
                artifact_file=f"student-viz/{full_tag}.png",
            )
            plt.cla()


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Initialize mlflow
    utils.mlflow_init()
    mlflow.set_experiment("harmonics-v3")

    with mlflow.start_run():
        # Log current script
        mlflow.log_artifact(local_path=pathlib.Path(__file__))

        # Log experiment cfg
        mlflow.log_params(dataclasses.asdict(cfg))

        # Run experiment
        run_experiment(cfg)


if __name__ == "__main__":
    main()

import dataclasses
import pathlib
import pickle
import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import src.utils as utils
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


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 2

    freq_limit: int = 4  # For ground truth harmonic function
    num_components: int = 8  # For ground truth harmonic function

    # (96, 192, 1) is from https://arxiv.org/pdf/2102.06701.pdf.
    layer_widths: tuple[int, ...] = (96, 192, 1)  # 1024, 42, 5, 1)

    n_val: int = 10000
    val_seed: int = -2

    train_sizes: tuple[int, ...] = tuple(int(2 ** x) for x in range(19))
    trials_per_size: int = 5

    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 16
    early_stopping_monitor: str = "val_mse"

    learning_rate: float = 3e-3
    max_epochs: int = 9001
    batch_size: int = 256
    l2_reg: float = 0

    num_workers: int = 0


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


@dataclasses.dataclass(frozen=True)
class TrainResult:
    # Final loss values
    final_train_mse: float
    final_val_mse: float
    final_epochs: int


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

    trainer = pl.Trainer(
        gpus=1,
        deterministic=True,
        enable_checkpointing=False,
        logger=False,  # We do custom logging instead.
        log_every_n_steps=1,
        max_epochs=cfg.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                patience=cfg.early_stopping_patience,
                mode="min",
            ),
            cc,
        ],
        enable_progress_bar=False,
        weights_summary=None,
    )
    trainer.fit(
        model=student,
        datamodule=dm,
    )

    def _get_mse(dl: DataLoader):
        (test_dict,) = trainer.test(
            model=student,
            dataloaders=dl,
            verbose=False,
        )
        return test_dict["test_mse"]

    return student, TrainResult(
        final_train_mse=_get_mse(dm.train_dataloader(shuffle=False)),
        final_val_mse=_get_mse(dm.val_dataloader()),
        final_epochs=cc.train_epochs_completed,
    )


def run_experiment(cfg: ExperimentConfig):
    assert all(perfect_pow_2(n) for n in cfg.train_sizes)

    hf = HarmonicFn(
        cfg=HarmonicFnConfig(
            input_dim=cfg.input_dim,
            freq_limit=cfg.freq_limit,
            num_components=cfg.num_components,
            seed=-1,
        )
    )
    if hf.cfg.input_dim == 2:
        hf.viz_2d(side_samples=512)
        mlflow.log_figure(figure=plt.gcf(), artifact_file="hf-viz.png")
        plt.cla()
    with open(mlflow.get_artifact_uri("hf.pkl"), "wb") as f:
        pickle.dump(hf, f)

    for n_train in cfg.train_sizes:
        for trial in range(cfg.trials_per_size):
            print(f"Training for {n_train=}, {trial=}")
            n_tag = f"n_{n_train:09}"
            full_tag = f"n_{n_train:09}_trial_{trial:02}"

            student, tr = train_student(
                hf=hf,
                n_train=n_train,
                cfg=cfg,
                tag=n_tag,
                seed=trial,
            )

            def _add_log10s(d: dict[str, float]) -> dict[str, float]:
                return d | {f"log10_{k}": np.log(v + 1e-42) for k, v in d.items()}

            print(tr)
            mlflow.log_metrics(
                _add_log10s(
                    dict(
                        train_mse=tr.final_train_mse,
                        val_mse=tr.final_val_mse,
                        epochs=tr.final_epochs,
                    )
                ),
                step=exact_log_2(n_train),
            )

            if student.cfg.input_dim == 2:
                student.viz_2d(side_samples=512)
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
    mlflow.set_experiment("harmonics")

    with mlflow.start_run():
        # Log current script
        mlflow.log_artifact(local_path=pathlib.Path(__file__))

        # Log experiment cfg
        mlflow.log_params(dataclasses.asdict(cfg))

        # Run experiment
        run_experiment(cfg)


if __name__ == "__main__":
    main()

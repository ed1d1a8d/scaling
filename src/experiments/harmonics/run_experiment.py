import dataclasses
import pathlib
import pickle
import warnings

import matplotlib.pyplot as plt
import mlflow
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

    freq_limit: int = 16  # For ground truth harmonic function
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


@dataclasses.dataclass(frozen=True)
class TrainResult:
    # Final loss values
    final_train_mse: float
    final_val_mse: float


def train_student(
    hf: HarmonicFn,
    n_train: int,
    cfg: ExperimentConfig,
    seed: int,
) -> TrainResult:
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

    trainer = pl.Trainer(
        gpus=1,
        deterministic=True,
        enable_checkpointing=False,
        max_epochs=cfg.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                patience=cfg.early_stopping_patience,
                mode="min",
            )
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

    return TrainResult(
        final_train_mse=_get_mse(dm.train_dataloader(shuffle=False)),
        final_val_mse=_get_mse(dm.val_dataloader()),
    )


def run_experiment(cfg: ExperimentConfig):
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

    for n_train_idx, n_train in enumerate(cfg.train_sizes):
        for trial in range(cfg.trials_per_size):
            print(f"Training for {n_train=}, {trial=}")
            tr = train_student(
                hf=hf,
                n_train=n_train,
                cfg=cfg,
                seed=trial,
            )

            print(tr)
            exit(0)

            """
            train_mse = float(hist.history["loss"][-1])
            val_mse = float(hist.history["val_loss"][-1])
            n_epochs = len(hist.history["loss"])
            print(f"train_mse ={train_mse}")
            print(f"val_mse   ={val_mse}")
            print(f"n_epochs  ={n_epochs}")
            mlflow.log_metrics(
                dict(train_mse=train_mse, val_mse=val_mse, n_epochs=n_epochs),
                step=n_idx,
            )

            if student.input_dim == 2:
                student.viz_2d(side_samples=512)
                mlflow.log_figure(
                    figure=plt.gcf(),
                    artifact_file=f"student-viz/n-{n:09}-trial-{trial:04}.png",
                )
                plt.cla()

            with open(mlflow.get_artifact_uri("histories.pkl"), "wb") as f:
                pickle.dump(
                    obj={k: [h.history for h in hs] for k, hs in histories.items()},
                    file=f,
                )
            """


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

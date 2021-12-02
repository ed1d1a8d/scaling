import dataclasses
import pathlib
import pickle
from collections import defaultdict

import elegy as eg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import optax
import src.models.mlp as mlp
import src.utils as utils
from simple_parsing import ArgumentParser
from src.experiments.harmonics.harmonic_function import HarmonicFunction


@dataclasses.dataclass
class ExperimentConfig:
    input_dim: int = 2

    freq_limit: int = 16  # For ground truth harmonic function
    num_components: int = 8  # For ground truth harmonic function

    # (96, 192, 1) is from https://arxiv.org/pdf/2102.06701.pdf.
    layer_widths: tuple[int, ...] = (96, 192, 1)  # 1024, 42, 5, 1)

    n_test: int = 10000
    ds_test_seed: int = -2

    train_sizes: tuple[int, ...] = tuple(int(2 ** x) for x in range(19))
    trials_per_size: int = 4

    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 32
    early_stopping_monitor: str = "val_mean_squared_error_loss"

    learning_rate: float = 3e-4
    max_epochs: int = 9001
    batch_size: int = 1024
    l2_reg: float = 0


def train_student(
    hf: HarmonicFunction,
    student_mod: eg.Module,
    n_train_samples: int,
    ds_test: dict[str, jnp.ndarray],
    cfg: ExperimentConfig,
    seed: int = 42,
    verbose: int = 0,
) -> eg.callbacks.History:
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(seed))

    student = eg.Model(
        module=student_mod,
        seed=rng1[0].item(),
        loss=[
            eg.losses.MeanSquaredError(),
            eg.regularizers.L2(l=cfg.l2_reg),
        ],
        optimizer=optax.adam(cfg.learning_rate),
    )

    ds_train = hf.get_iid_dataset(
        n_samples=n_train_samples,
        batch_size=cfg.batch_size,
        rng=rng2,
    )

    return student.fit(
        inputs=ds_train["xs"],
        labels=ds_train["ys"],
        validation_data=(ds_test["xs"], ds_test["ys"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        epochs=cfg.max_epochs,
        callbacks=[
            eg.callbacks.EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                patience=cfg.early_stopping_patience,
                mode="min",
            )
        ]
        if cfg.early_stopping
        else [],
        verbose=verbose,
        drop_remaining=False,
    )


def run_experiment(cfg: ExperimentConfig):
    hf = HarmonicFunction(
        input_dim=cfg.input_dim,
        freq_limit=cfg.freq_limit,
        num_components=cfg.num_components,
        seed=-1,
    )
    if hf.input_dim == 2:
        hf.viz_2d(side_samples=512)
        mlflow.log_figure(figure=plt.gcf(), artifact_file="teacher-viz.png")
        plt.cla()
    with open(mlflow.get_artifact_uri("hf.pkl"), "wb") as f:
        pickle.dump(hf, f)
    ds_test = hf.get_iid_dataset(
        n_samples=cfg.n_test,
        rng=jax.random.PRNGKey(-2),
        batch_size=cfg.batch_size,
    )

    student_mod = mlp.MLP(
        input_dim=cfg.input_dim,
        layer_widths=cfg.layer_widths,
    )

    histories: dict[int, list[eg.callbacks.History]] = defaultdict(list)

    for n_idx, n in enumerate(cfg.train_sizes):
        keys = jax.random.split(jax.random.PRNGKey(n), cfg.trials_per_size)
        for trial, key in enumerate(keys):
            print(f"Training for {n=}, {trial=}")
            hist = train_student(
                hf=hf,
                student_mod=student_mod,
                n_train_samples=n,
                ds_test=ds_test,
                seed=key[0],
                cfg=cfg,
            )

            histories[n].append(hist)

            train_mse = float(hist.history["mean_squared_error_loss"][-1])
            val_mse = float(hist.history["mean_squared_error_loss"][-1])
            print(f"train_mse={train_mse}")
            print(f"val_mse  ={val_mse}")
            mlflow.log_metrics(
                dict(train_mse=train_mse, val_mse=val_mse),
                step=n_idx,
            )

            utils.mlflow_log_jax(
                {k: [h.history for h in hs] for k, hs in histories.items()},
                artifact_name="histories.bin",
            )


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

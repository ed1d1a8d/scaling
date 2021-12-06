import dataclasses
import pathlib
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import mlflow
import src.models.mlp as mlp
import src.utils as utils
from simple_parsing import ArgumentParser
from src.experiments.harmonics.harmonic_function import HarmonicFunction


_tf_config = tf.compat.v1.ConfigProto()
_tf_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=_tf_config)


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
    trials_per_size: int = 5

    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 16
    early_stopping_monitor: str = "val_loss"

    learning_rate: float = 3e-3
    max_epochs: int = 9001
    batch_size: int = 256
    l2_reg: float = 0


def train_student(
    hf: HarmonicFunction,
    n_train_samples: int,
    ds_test: dict[str, tf.Tensor],
    cfg: ExperimentConfig,
    rng: tf.random.Generator,
    verbose: int = 0,
) -> tuple[mlp.MLP, keras.callbacks.History]:
    student = mlp.MLP(
        input_dim=cfg.input_dim,
        layer_widths=cfg.layer_widths,
        seed=rng.make_seeds(1)[0][0].numpy(),
    )
    student.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss=keras.losses.MeanSquaredError(),  # TODO: Add regularization support
    )

    ds_train = hf.get_iid_dataset(
        n_samples=n_train_samples,
        batch_size=cfg.batch_size,
        seed=rng.make_seeds(1)[0][0].numpy(),
    )

    hist = student.fit(
        x=ds_train["xs"],
        y=ds_train["ys"],
        validation_data=(ds_test["xs"], ds_test["ys"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        epochs=cfg.max_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                patience=cfg.early_stopping_patience,
                mode="min",
            )
        ]
        if cfg.early_stopping
        else [],
        verbose=verbose,
    )

    return student, hist


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
        seed=-2,
        batch_size=cfg.batch_size,
    )

    histories: dict[int, list[keras.callbacks.History]] = defaultdict(list)

    for n_idx, n in enumerate(cfg.train_sizes):
        rngs = tf.random.Generator.from_seed(n).split(count=cfg.trials_per_size)
        for trial, rng in enumerate(rngs):
            print(f"Training for {n=}, {trial=}")
            student, hist = train_student(
                hf=hf,
                n_train_samples=n,
                ds_test=ds_test,
                rng=rng,
                cfg=cfg,
            )

            histories[n].append(hist)

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

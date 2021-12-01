import dataclasses
import pathlib
from collections import defaultdict

import elegy as eg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import optax
import src.experiments.mlp_scaling.mlp as mlp
import src.utils as utils


@dataclasses.dataclass
class ExperimentConfig:
    input_dim: int = 8
    layer_widths: tuple[int, ...] = (96, 192, 1)  # 1024, 42, 5, 1)

    n_test: int = 1024
    ds_test_seed: int = -2

    train_sizes: tuple[int] = tuple(int(1.7 ** x) for x in range(1, 25))
    trials_per_size: int = 4

    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 16
    early_stopping_monitor: str = "val_mean_squared_error_loss"

    learning_rate: float = 3e-2
    max_epochs: int = 512
    batch_size: int = 256
    l2_reg: float = 0


def train_student(
    student_mod: eg.Module,
    teacher: eg.Model,
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
            # mlp.BandwidthLoss(),
            eg.regularizers.L2(l=cfg.l2_reg),
        ],
        optimizer=optax.adam(cfg.learning_rate),
    )

    ds_train = mlp.get_iid_dataset(
        model=teacher,
        n_samples=n_train_samples,
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
    teacher_mod = mlp.MLP(
        input_dim=cfg.input_dim,
        layer_widths=cfg.layer_widths,
    )
    student_mod = mlp.MLP(
        input_dim=cfg.input_dim,
        layer_widths=cfg.layer_widths,
    )

    teacher = mlp.get_random_mlp(mlp=teacher_mod, seed=-1)
    teacher.save(path=mlflow.get_artifact_uri("teacher"))
    if teacher_mod.input_dim == 2:
        mlp.viz_model(teacher, 512)
        mlflow.log_figure(figure=plt.gcf(), artifact_file="teacher-viz.png")
        plt.cla()

    ds_test = mlp.get_iid_dataset(
        model=teacher,
        n_samples=cfg.n_test,
        rng=jax.random.PRNGKey(-2),
    )

    histories: dict[int, list[eg.callbacks.History]] = defaultdict(list)

    for n in cfg.train_sizes:
        keys = jax.random.split(jax.random.PRNGKey(n), cfg.trials_per_size)
        for key in keys:
            print(f"Training for {n=}")
            hist = train_student(
                student_mod=student_mod,
                teacher=teacher,
                n_train_samples=n,
                ds_test=ds_test,
                seed=key[0],
                cfg=cfg,
            )

            histories[n].append(hist)

            print(f"train_mse={hist.history['mean_squared_error_loss'][-1]}")
            print(f"val_mse  ={hist.history['val_mean_squared_error_loss'][-1]}")

            utils.mlflow_log_jax(
                {k: [h.history for h in hs] for k, hs in histories.items()},
                artifact_name="histories.bin",
            )


def main():
    utils.mlflow_init()
    mlflow.set_experiment("mlp-student-teacher")

    cfg = ExperimentConfig()

    with mlflow.start_run():
        # Log current script
        mlflow.log_artifact(local_path=pathlib.Path(__file__))

        # Log experiment cfg
        mlflow.log_params(dataclasses.asdict(cfg))

        # Run experiment
        run_experiment(cfg)


if __name__ == "__main__":
    main()

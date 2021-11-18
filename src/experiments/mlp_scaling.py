from dataclasses import dataclass

import elegy as eg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax


class MLP(eg.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        layer_widths: tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_widths = layer_widths
        self.input_shape = input_shape

    @eg.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for w in self.layer_widths[:-1]:
            x = eg.Linear(
                w,
                kernel_init=eg.initializers.he_normal(),
            )(x)
            x = jax.nn.relu(x)

        x = eg.Linear(
            self.layer_widths[-1],
            kernel_init=eg.initializers.he_normal(),
        )(x)
        return x


def get_random_mlp(
    mlp: MLP,
    seed: int = 0,
) -> eg.Model:
    """
    [96, 192, 1] is from https://arxiv.org/pdf/2102.06701.pdf.
    """
    model = eg.Model(module=mlp, seed=seed)

    assert not model.initialized
    model.predict(jnp.zeros(mlp.input_shape))
    assert model.initialized

    return model


def sq_sampling(side_samples: int):
    xs, ys = jnp.meshgrid(
        jnp.linspace(-1, 1, side_samples),
        jnp.linspace(-1, 1, side_samples),
    )
    return jnp.stack([xs, ys]).T


def get_model_img(
    model: eg.Model,
    side_samples: int,
) -> jnp.ndarray:
    XY = sq_sampling(side_samples)
    assert XY.shape == (side_samples, side_samples, 2)

    return model.predict(XY).squeeze(-1)


def viz_model(
    model: eg.Model,
    side_samples: int,
) -> None:
    img = get_model_img(model=model, side_samples=side_samples)
    plt.imshow(img)


def get_iid_dataset(
    model: eg.Model,
    n_samples: int,
    rng: jax.random.KeyArray,
    batch_size: int = 256,
) -> dict[str, jnp.ndarray]:
    xs = jax.random.uniform(
        minval=-1,
        maxval=1,
        shape=(n_samples,) + model.module.input_shape,
        key=rng,
    )

    ys = model.predict(xs, batch_size=batch_size)

    return dict(xs=xs, ys=ys)


@dataclass
class TrainResult:
    model: eg.Model
    history: eg.callbacks.History


def train_student(
    student_mod: eg.Module,
    teacher: eg.Model,
    n_train_samples: int,
    ds_test: dict[str, jnp.ndarray],
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    max_epochs: int = 512,
    seed: int = 42,
    verbose: int = 0,
) -> TrainResult:
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(seed))

    student = eg.Model(
        module=student_mod,
        seed=rng1[0].item(),
        loss=[
            eg.losses.MeanSquaredError(),
            # eg.regularizers.GlobalL2(l=1e-4),
        ],
        optimizer=optax.adam(learning_rate),
    )

    ds_train = get_iid_dataset(
        model=teacher,
        n_samples=n_train_samples,
        rng=rng2,
    )

    history = student.fit(
        inputs=ds_train["xs"],
        labels=ds_train["ys"],
        validation_data=(ds_test["xs"], ds_test["ys"]),
        batch_size=batch_size,
        shuffle=True,
        epochs=max_epochs,
        callbacks=[eg.callbacks.EarlyStopping(monitor="val_loss", patience=3)],
        verbose=verbose,
        drop_remaining=False,
    )

    return TrainResult(model=student, history=history)

import typing as tp

import elegy as eg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class MLP(eg.Module):
    def __init__(
        self,
        input_dim: int,
        layer_widths: tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_widths = layer_widths
        self.input_dim = input_dim

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
    model.predict(jnp.zeros(mlp.input_dim))
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
    # TODO: Extend to more than 2 dimensions
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
        shape=(n_samples, model.module.input_dim),
        key=rng,
    )

    ys = model.predict(xs, batch_size=batch_size)

    return dict(xs=xs, ys=ys)


class BandwidthLoss(eg.Loss):
    def __init__(
        self,
        side_samples: int,
        weight: tp.Optional[float] = None,
        name: tp.Optional[str] = None,
    ):
        """
        side_samples: Number of samples per side used to compute the DFT.
            Total number of samples is side_samples ** input_dim.
        weight: Optional weight contribution for the total loss. Defaults to `1`.
        name: Optional name for the instance, if not provided lower snake_case version
            of the name of the class is used instead.
        """
        self.side_samples = side_samples
        self.name = name if name is not None else eg.utils.get_name(self)
        self.weight = weight if weight is not None else 1.0

    def call(self, states: eg.Model):
        # Fn arg needs to be named states to conform with elegy API.
        # We rename it to `model` make things more clear later on.
        model = states

        img = get_model_img(
            model=model,
            side_samples=self.side_samples,
        )

        return 0

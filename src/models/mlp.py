"""
Implementation of a multilayer perceptron (aka a fully connected net) in elegy.
Also contains useful utilities.
"""

import elegy as eg
import jax
import jax.numpy as jnp


class MLP(eg.Module):
    def __init__(
        self,
        input_dim: int,
        layer_widths: tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.layer_widths = layer_widths

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
    """Returns a randomly initialized MLP."""
    model = eg.Model(module=mlp, seed=seed)

    assert not model.initialized
    model.predict(jnp.zeros(mlp.input_dim))
    assert model.initialized

    return model

"""Code for representing harmonic functions."""

import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class HarmonicFunction:
    """
    Represents a bandlimited function with of a fixed number of
    integer-frequency harmonics.
    The function is scaled so as to be periodic on the unit-square.
    """

    def __init__(
        self,
        input_dim: int,
        freq_limit: int,
        num_components: int,
        seed: int = 42,
    ):
        """Creates a random harmonic function."""
        rng1, rng2 = jax.random.split(key=jax.random.PRNGKey(seed))

        self.input_dim = input_dim
        self.freq_limit = freq_limit
        self.num_components = num_components

        self.coeffs = jax.random.normal(key=rng1, shape=(num_components,))
        self.freqs = jax.random.randint(
            key=rng2,
            minval=0,
            maxval=freq_limit + 1,
            shape=(num_components, input_dim),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict(self, xs: jnp.ndarray) -> jnp.ndarray:
        assert xs.shape[-1] == self.input_dim
        # return jnp.exp(2j * jnp.pi * (xs @ self.freqs.T)) @ self.coeffs
        return jnp.cos(2 * jnp.pi * (xs @ self.freqs.T)) @ self.coeffs

    def predict(
        self,
        xs: jnp.ndarray,
        batch_size: int = 32,
    ) -> jnp.ndarray:
        batches = jnp.split(xs, jnp.arange(batch_size, xs.shape[0], batch_size))
        ys = jnp.concatenate(
            [self._predict(batch) for batch in batches],
        )
        return ys

    def get_iid_dataset(
        self,
        n_samples: int,
        rng: jax.random.KeyArray,
        batch_size: int = 2048,
    ) -> dict[str, jnp.ndarray]:
        """
        iid dataset of (x, hf(x)) with x sampled uniformly over the unit square.
        """
        xs = jax.random.uniform(
            minval=0,
            maxval=1,
            shape=(n_samples, self.input_dim),
            key=rng,
        )
        ys = self.predict(xs, batch_size=batch_size)

        return dict(xs=xs, ys=ys)

    def viz_2d(self, side_samples: int):
        assert self.input_dim == 2
        s = slice(0, 1, 1j * side_samples)
        XY = jnp.mgrid[s, s].T
        img = self._predict(XY)
        plt.imshow(img, origin="lower")

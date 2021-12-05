"""Code for representing harmonic functions."""

import math

import jax.numpy as jnp
import src.utils as utils
import tensorflow as tf


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
        rng = tf.random.Generator.from_seed(seed)

        self.input_dim = input_dim
        self.freq_limit = freq_limit
        self.num_components = num_components

        self.coeffs = tf.linalg.normalize(
            rng.normal(shape=(num_components,)),
        )[0]

        self.freqs = tf.cast(
            rng.uniform(
                shape=(num_components, input_dim),
                minval=0,
                maxval=freq_limit + 1,
                dtype=tf.int32,
            ),
            tf.float32,
        )

    def _predict(self, xs: tf.Tensor) -> tf.Tensor:
        assert xs.shape[-1] == self.input_dim

        # Could also do exp
        return tf.linalg.matvec(
            tf.cos(2 * math.pi * xs @ tf.transpose(self.freqs)),
            self.coeffs,
        )

    def predict(
        self,
        xs: tf.Tensor,
        batch_size: int = 32,
    ) -> tf.Tensor:
        n = len(xs)
        batches = tf.split(
            xs,
            [batch_size for _ in range(n // batch_size)]
            + ([] if n % batch_size == 0 else [n % batch_size]),
        )

        ys = tf.concat([self._predict(batch) for batch in batches], axis=0)
        return ys

    def get_iid_dataset(
        self,
        n_samples: int,
        seed: int = 42,
        batch_size: int = 2048,
    ) -> dict[str, tf.Tensor]:
        """
        iid dataset of (x, hf(x)) with x sampled uniformly over the unit square.
        """
        rng = tf.random.Generator.from_seed(seed)
        xs = rng.uniform(
            minval=0,
            maxval=1,
            shape=(n_samples, self.input_dim),
        )
        ys = self.predict(xs, batch_size=batch_size)

        return dict(xs=xs, ys=ys)

    def viz_2d(self, side_samples: int):
        assert self.input_dim == 2
        utils.viz_2d(
            lambda xs: jnp.array(self._predict(tf.convert_to_tensor(xs))),
            side_samples=side_samples,
        )

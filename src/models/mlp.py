"""
Implementation of a multilayer perceptron (aka a fully connected net) in tf.
Also contains useful utilities.
"""

import tensorflow as tf
import tensorflow.keras as keras


class MLP(keras.Model):
    def __init__(
        self,
        input_dim: int,
        layer_widths: tuple[int, ...],
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.layer_widths = layer_widths

        tf.random.set_seed(seed)
        self.net = keras.Sequential(
            [keras.Input(shape=(input_dim,))]
            + [keras.layers.Dense(w, activation="relu") for w in layer_widths[:-1]]
            + [keras.layers.Dense(layer_widths[-1])]
        )

    def __call__(self, xs: tf.Tensor) -> tf.Tensor:
        return self.net(xs)

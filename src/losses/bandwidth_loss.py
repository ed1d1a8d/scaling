"""Implements various losses for regularizing a functions bandwidth."""

from typing import Optional

import elegy as eg
import jax.numpy as jnp


def grid_sampled_cube(
    side_samples: int,
    dimensions: int,
) -> jnp.ndarray:
    """
    Example:
        When side_samples=10 and dimensions=3
        returns an array of shape (10, 10, 10, 3).
    """
    ret = jnp.moveaxis(
        jnp.mgrid[(slice(0, 1, 1j * side_samples) for _ in range(dimensions))],
        source=0,
        destination=-1,
    )
    assert ret.shape == tuple(side_samples for _ in range(dimensions)) + (dimensions,)
    return ret


def model_on_grid_sampled_cube(
    model: eg.Model,
    side_samples: int,
) -> jnp.ndarray:
    samples = grid_sampled_cube(side_samples, model.module.input_dim)
    return model.predict(samples).squeeze(-1)


class SampledBandwidthLoss(eg.Loss):
    """
    A loss that regularizes a parameterized functions bandwidth according to
    the DFT/DCT basis computed from a sampled version of the function.

    The domain of the function is assumed to be [0, 1]^d
    (i.e. the unit hypercube).
    """

    def __init__(
        self,
        side_samples: int,
        weight: Optional[float] = None,
        name: Optional[str] = None,
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

        vals_on_cube = model_on_grid_sampled_cube(
            model=model,
            side_samples=self.side_samples,
        )

        raise NotImplementedError

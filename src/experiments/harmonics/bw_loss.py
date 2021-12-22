"""Losses for regularizing a function's bandwidth."""

from typing import Callable, Optional

import numpy as np
import torch


def high_freq_norm_mc(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int,
    bandlimit: int,
    n_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes the 2-norm of high frequency fourier components of `fn` using
    Monte-Carlo integration with `n_samples` samples.

    High frequency is defined as frequencies with L_infty norm above
    `bandlimit`.

    `fn` is assumed to be periodic over the unit hypercube.
    """
    BASIS_SZ = 2 * ((bandlimit + 1) ** input_dim) - 1
    if n_samples <= BASIS_SZ:
        return 0

    xs = torch.rand((n_samples, input_dim), device=device)
    ys = fn(xs)

    basis_freqs = torch.Tensor(
        np.mgrid[tuple(slice(0, bandlimit + 1) for _ in range(input_dim))]
        .reshape(input_dim, -1)
        .T
    ).to(device)
    basis_ys: torch.Tensor = torch.concat(
        (
            torch.cos(2 * np.pi * basis_freqs @ xs.T),
            torch.sin(2 * np.pi * basis_freqs @ xs.T)[1:],
        ),
        axis=0,
    )
    basis_ys /= torch.sqrt((basis_ys * basis_ys).mean(axis=-1, keepdim=True))
    assert basis_ys.shape == (BASIS_SZ, n_samples)

    residuals = basis_ys.T @ (torch.pinverse(basis_ys.T) @ ys) - ys
    return (residuals * residuals).mean()

    # Alt. Method 1:
    # Treat basis_ys as having approximately orthogonal columns.
    # approx_basis_coeffs = basis_ys @ ys / n_samples
    # residuals = approx_basis_coeffs @ basis_ys - ys
    # return (residuals * residuals).mean()

    # Alt. Method 2:
    # Alternate implementation using torch.linalg.lstsq doesn't work because
    # lstsq is not differentiable...
    #
    #   res: torch.Tensor = torch.linalg.lstsq(basis_ys, ys).residuals[0]
    #   assert res.shape == ()
    #
    # See https://github.com/pytorch/pytorch/issues/27036#issuecomment-743413633
    # for details.


def high_freq_norm_dft(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int,
    bandlimit: int,
    side_samples: int,
    device: Optional[torch.device] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Computes the 2-norm of high frequency fourier components of `fn` using
    a discrete fourier transform with side_samples per side.

    High frequency is defined as frequencies with L_infty norm above
    `bandlimit`.

    `fn` is assumed to be periodic over the unit hypercube.
    """
    if 2 * bandlimit + 1 >= side_samples:
        return 0

    grid_xs = torch.Tensor(
        np.moveaxis(
            np.mgrid[tuple(slice(0, side_samples) for _ in range(input_dim))],
            source=0,
            destination=-1,
        )
        / side_samples,
        device=device,
    )
    grid_ys = fn(grid_xs)

    grid_fft = torch.fft.fftn(grid_ys, norm="forward")
    grid_sfft = torch.fft.fftshift(grid_fft)

    high_freq_coeffs = grid_sfft
    mid_idx = side_samples // 2

    high_freq_coeffs[
        tuple(slice(mid_idx, mid_idx + bandlimit + 1) for _ in range(input_dim))
    ] = 0
    high_freq_coeffs[
        tuple(slice(mid_idx - bandlimit, mid_idx + 1) for _ in range(input_dim))
    ] = 0

    return high_freq_coeffs.norm() ** 2

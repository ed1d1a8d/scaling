"""Losses for regularizing a function's bandwidth."""

from typing import Callable, Optional

import numpy as np
import torch


def high_freq_norm_mcls(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int,
    freq_limit: int,
    n_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes the 2-norm of high frequency fourier components of `fn` using
    Monte-Carlo least-squares with `n_samples` samples.

    High frequency is defined as frequencies with L_infty norm above
    `freq_limit`.

    `fn` is assumed to be periodic over the unit hypercube.
    """
    BASIS_SZ = (2 * freq_limit + 1) ** input_dim
    if n_samples <= BASIS_SZ:
        return torch.tensor(0)

    xs = torch.rand((n_samples, input_dim), device=device)
    ys = fn(xs)

    all_freqs = (
        np.mgrid[tuple(slice(-freq_limit, freq_limit + 1) for _ in range(input_dim))]
        .reshape(input_dim, -1)
        .T
    )

    def _include_freq(f: np.ndarray) -> bool:
        nonzero_coords = f[f != 0]
        if len(nonzero_coords) == 0:
            return True
        return nonzero_coords[0] > 0

    basis_freqs = torch.Tensor(
        np.stack(
            [f for f in all_freqs if _include_freq(f)],
            axis=0,
        )
    ).to(device)

    basis_ys: torch.Tensor = torch.concat(
        (
            torch.cos(2 * np.pi * basis_freqs @ xs.T),
            torch.sin(2 * np.pi * basis_freqs @ xs.T)[1:],
        ),
        dim=0,
    )
    basis_ys /= torch.sqrt((basis_ys * basis_ys).mean(dim=-1, keepdim=True))
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
    freq_limit: int,
    side_samples: int,
    device: Optional[torch.device] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Computes the 2-norm of high frequency fourier components of `fn` using
    a discrete fourier transform with side_samples per side.

    High frequency is defined as frequencies with L_infty norm above
    `freq_limit`.

    `fn` is assumed to be periodic over the unit hypercube.
    """
    if 2 * freq_limit + 1 >= side_samples:
        return torch.tensor(0)

    grid_xs = torch.Tensor(
        np.moveaxis(
            np.mgrid[tuple(slice(0, side_samples) for _ in range(input_dim))],
            source=0,
            destination=-1,
        )
        / side_samples,
    ).to(device)
    grid_ys = fn(grid_xs)

    grid_fft = torch.fft.fftn(grid_ys, norm="forward")
    grid_sfft = torch.fft.fftshift(grid_fft)

    high_freq_coeffs = grid_sfft
    mid_idx = side_samples // 2

    high_freq_coeffs[
        tuple(
            slice(mid_idx - freq_limit, mid_idx + freq_limit + 1)
            for _ in range(input_dim)
        )
    ] = 0

    return high_freq_coeffs.norm() ** 2

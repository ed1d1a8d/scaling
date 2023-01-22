"""Functional forms for scaling laws."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseLaw(ABC, nn.Module):
    """Base class for a scaling law."""

    @abstractmethod
    def forward(
        self,
        n_params: torch.Tensor,
        n_trains: torch.Tensor,
    ) -> torch.Tensor:
        """Returns log L(n_params, n_trains)"""

        raise NotImplementedError


class BasicPowerLaw(BaseLaw):
    """
    L(n_param, n_train) = e^a0 * n_param^{-a1} + e^b0 * n_train^{-b1} + e^c
    """

    def __init__(
        self,
        a0: float,
        a1: float,
        b0: float,
        b1: float,
        c: float,
    ):
        super().__init__()
        self.a0 = nn.Parameter(torch.tensor(a0, dtype=torch.float32))  # type: ignore
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))  # type: ignore
        self.b0 = nn.Parameter(torch.tensor(b0, dtype=torch.float32))  # type: ignore
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32))  # type: ignore
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))  # type: ignore

    def forward(
        self,
        n_params: torch.Tensor,
        n_trains: torch.Tensor,
    ) -> torch.Tensor:
        """Returns log L(n_params, n_trains)"""

        return torch.logsumexp(
            torch.stack(
                (
                    self.a0 - self.a1 * torch.log(n_params),
                    self.b0 - self.b1 * torch.log(n_trains),
                    0 * n_params + self.c,
                )
            ),
            dim=0,
        )


class InterpPowerLaw(BaseLaw):
    """
    L_interp(n_param, n_train) =
        e^eps0
        * L_base(n_param, n_train)
        / sqrt(L_base(n_param, n_train)^2 + e^gamma)

    This is equivalent to the functional form in
    https://arxiv.org/pdf/1909.12673.pdf.
    """

    def __init__(
        self,
        base_power_law: BasicPowerLaw,
        eps0: float,
        gamma: float,
    ):
        super().__init__()
        self.base_power_law = base_power_law
        self.eps0 = nn.Parameter(torch.tensor(eps0, dtype=torch.float32))  # type: ignore
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))  # type: ignore

    def forward(
        self,
        n_params: torch.Tensor,
        n_trains: torch.Tensor,
    ) -> torch.Tensor:
        """Returns log L_interp(n_params, n_trains)"""

        log_base = self.base_power_law(n_params, n_trains)
        return (
            self.eps0
            + log_base
            - 0.5
            * torch.logsumexp(
                torch.stack((2 * log_base, self.gamma + n_params * 0)), dim=0
            )
        )


class GaussianLaw(BaseLaw):
    """
    L_interp(n_param, n_train) = Phi(
        - e^s
        / sqrt(
            1 + e^{-2s} * n_param /
                (e^b0 * n_train^{b1})
        )
    )
    """

    def __init__(
        self,
        s: float,
        b0: float,
        b1: float,
    ):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))  # type: ignore
        self.b0 = nn.Parameter(torch.tensor(b0, dtype=torch.float32))  # type: ignore
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32))  # type: ignore

    def forward(
        self,
        n_params: torch.Tensor,
        n_trains: torch.Tensor,
    ) -> torch.Tensor:
        """Returns log L_interp(n_params, n_trains)"""

        return torch.special.log_ndtr(
            -torch.exp(self.s)
            / torch.sqrt(
                1
                + n_params
                * torch.exp(
                    -2 * self.s - self.b0 - self.b1 * torch.log(n_trains)
                )
            )
        )

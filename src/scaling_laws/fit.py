from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.scaling_laws.ff import BaseLaw


def _fit_law(
    model: BaseLaw,
    n_params: np.ndarray,
    n_trains: np.ndarray,
    errs: np.ndarray,
    max_iters: int = 20,
    huber_delta: float = 1e-3,
    show_progress: bool = False,
):
    n_params_t = torch.tensor(n_params)
    n_trains_t = torch.tensor(n_trains)
    errs_t = torch.tensor(errs)

    # Optimize model using LBFGS
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=max_iters,
    )

    def closure():
        optimizer.zero_grad()
        pred_log_errs = model(n_params=n_params_t, n_trains=n_trains_t)
        loss = nn.functional.huber_loss(
            pred_log_errs,
            torch.log(errs_t),
            reduction="mean",
            delta=huber_delta,
        )
        loss.backward()
        return loss

    losses = []
    pbar = tqdm(range(max_iters), disable=not show_progress)
    for _ in pbar:
        loss = optimizer.step(closure)  # type: ignore

        losses.append(loss.item())  # type: ignore
        pbar.set_description(f"Loss: {loss.item():.4e}")  # type: ignore

    return model, np.array(losses)


def fit_law(
    get_random_model: Callable[[], BaseLaw],
    n_params: np.ndarray,
    n_trains: np.ndarray,
    errs: np.ndarray,
    max_iters: int = 20,
    huber_delta: float = 1e-3,
    show_inner_progress: bool = False,
    seed: int = 42,
    n_random_inits=100,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    min_loss = np.inf
    best_losses: Optional[np.ndarray] = None
    best_model: Optional[BaseLaw] = None

    pbar = tqdm(range(n_random_inits))
    for _ in pbar:
        model = get_random_model()

        model, losses = _fit_law(
            model=model,
            n_params=n_params,
            n_trains=n_trains,
            errs=errs,
            max_iters=max_iters,
            huber_delta=huber_delta,
            show_progress=show_inner_progress,
        )
        if losses[-1] < min_loss:
            min_loss = losses[-1]
            best_model = model
            best_losses = losses

        pbar.set_description(f"Loss: {min_loss:.4e}")

    return best_model, best_losses

"""Minimal PyTorch PINN toolkit, shared by the module-03 notebooks.

Ported from the original JAX teaching code to PyTorch (the course environment).
Deliberately small and explicit:

    * `MLP`           -- a tanh multilayer perceptron (smooth, so the residual
                         can be differentiated to 2nd order; never use ReLU).
    * `grad`          -- convenience wrapper around torch.autograd.grad for
                         building PDE/ODE residuals.
    * `train`         -- a plain Adam loop with optional step-decay LR.
    * `set_plot_style`-- consistent figure styling.

Double precision is enabled by callers (`torch.set_default_dtype(torch.float64)`)
because the residual differentiates the network twice and float32 round-off
would dominate the physics loss.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Tanh MLP. `layers` e.g. [1, 32, 32, 1] = 1 input, two hidden, 1 output."""

    def __init__(self, layers: list[int]):
        super().__init__()
        mods: list[nn.Module] = []
        for d_in, d_out in zip(layers[:-1], layers[1:]):
            mods.append(nn.Linear(d_in, d_out))
            mods.append(nn.Tanh())
        mods.pop()  # drop the final activation -> linear output
        self.net = nn.Sequential(*mods)
        # Glorot/Xavier init, matching the original code.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y / d x with the graph retained (for higher-order derivatives)."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def train(loss_fn: Callable[[], torch.Tensor],
          params,
          steps: int = 5000,
          lr: float = 1e-3,
          decay_every: int = 2000,
          factor: float = 0.5,
          log_every: int = 0):
    """Adam optimization of `loss_fn` over `params` (an iterable of tensors).

    Returns the list of loss values. `params` may mix network parameters and
    bare trainable tensors (e.g. an unknown physical coefficient).
    """
    params = list(params)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=factor)
    history = []
    for k in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if log_every and (k % log_every == 0):
            print(f"step {k:6d}   loss {loss.item():.3e}")
    return history


# Styling moved to pyutils.plotstyle (single source of truth). Re-exported so
# existing `import pinn_core as pc; pc.set_plot_style()` call sites keep working.
from .plotstyle import save_dual, set_plot_style  # noqa: E402,F401

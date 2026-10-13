"""2-D heat-diffusion reference solver (explicit FTCS).

The course's first PDE benchmark:

    dT/dt = alpha * laplacian(T) + qdot / (rho cp)

solved with central differences in space and forward Euler in time (FTCS) on a
uniform grid. The explicit scheme is stable only under the von Neumann limit
returned by `stable_dt`. Used by the gluing, circuit-board and (1-D) egg
applications of Module 01 and by the 2-D surrogate of Module 02. The 1-D heat
rod that Modules 20-26 use is defined in their own notebooks (the CoSTA module
has its own backward-Euler solver); see Audit_Hybrid_Story.md for the plan to
share one definition.

Notation: scalar fields/properties italic (T, alpha, k); the flux is a vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class HeatProblem:
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 61
    ny: int = 61
    alpha: float = 1.0e-4          # thermal diffusivity k/(rho cp) [m^2/s]
    # Dirichlet boundary temperatures (None => insulated / zero-flux Neumann)
    T_left: float | None = 1.0
    T_right: float | None = 1.0
    T_bottom: float | None = 0.0
    T_top: float | None = 0.0
    T_init: float = 0.0
    # volumetric source term S = qdot/(rho cp) as S(X, Y, t); default none
    source: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None

    def grid(self):
        x = np.linspace(0.0, self.Lx, self.nx)
        y = np.linspace(0.0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return x, y, X, Y


def stable_dt(alpha: float, dx: float, dy: float, safety: float = 0.9) -> float:
    """von Neumann stability limit for 2-D FTCS (with a safety factor)."""
    return safety / (2.0 * alpha) * (dx**2 * dy**2) / (dx**2 + dy**2)


def _apply_bcs(T: np.ndarray, p: HeatProblem) -> None:
    """Apply Dirichlet (fixed T) or zero-flux Neumann (mirror) on each edge."""
    T[0, :] = p.T_left if p.T_left is not None else T[1, :]
    T[-1, :] = p.T_right if p.T_right is not None else T[-2, :]
    T[:, 0] = p.T_bottom if p.T_bottom is not None else T[:, 1]
    T[:, -1] = p.T_top if p.T_top is not None else T[:, -2]


def solve(p: HeatProblem, t_end: float, dt: float | None = None, save_every: int = 0):
    """Integrate the heat equation to t_end with FTCS.

    Returns (x, y, T) for the final field, or (x, y, T, frames, times) when
    save_every > 0 (frames = list of snapshots every `save_every` steps).
    """
    x, y, X, Y = p.grid()
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = dt or stable_dt(p.alpha, dx, dy)
    n = int(np.ceil(t_end / dt))

    T = np.full((p.nx, p.ny), float(p.T_init))
    _apply_bcs(T, p)
    frames, times = [], []
    a = p.alpha
    for k in range(n):
        t = k * dt
        lap = np.zeros_like(T)
        lap[1:-1, 1:-1] = (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2
            + (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        )
        S = p.source(X, Y, t) if p.source is not None else 0.0
        T = T + dt * (a * lap + S)
        _apply_bcs(T, p)
        if save_every and (k % save_every == 0):
            frames.append(T.copy())
            times.append((k + 1) * dt)

    if save_every:
        return x, y, T, frames, times
    return x, y, T


if __name__ == "__main__":
    p = HeatProblem(nx=41, ny=41)
    print("stable dt:", stable_dt(p.alpha, p.Lx / (p.nx - 1), p.Ly / (p.ny - 1)))
    x, y, T = solve(p, t_end=200.0)
    print("T range after solve:", float(T.min()), float(T.max()))

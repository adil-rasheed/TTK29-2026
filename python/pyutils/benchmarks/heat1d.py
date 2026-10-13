"""The canonical 1-D heat rod shared by Modules 20-26.

One definition so that every hybrid-modelling module solves the SAME problem:

    dT/dt = alpha * d2T/dx2 (+ q(x)),   x in [0, 1],   T(0, t) = T(1, t) = 0,
    alpha = 0.1,
    initial profiles from the eight-mode sine family
        a(x) = sum_{k=1}^{8} c_k sin(k pi x),   c_k ~ N(0, 1/k^2),
    whose first member (c = e_1) is the single problem of the PINN module.

The reference solver is the explicit FTCS march of Module 01 on NX = 128 points
with a ceiling-based step count so that it lands exactly on the requested time;
the analytic sine series is available to check it.  Module 25 (DeepONet) uses
the final time T_F = 0.05 (mode k keeps exp(-alpha k^2 pi^2 T_F): 0.95 for k = 1,
0.04 for k = 8); Module 22 (PINN) runs the first member to t = 1.

Notation: scalar fields italic (T, alpha); the coefficient vector c is a NumPy
array.  Nothing here depends on torch.
"""
from __future__ import annotations

import math

import numpy as np

ALPHA = 0.1          # diffusivity shared by Modules 22 and 25 (and the canonical rod)
T_F = 0.05           # final time of the operator examples
NX = 128             # points of the reference grid
NMODE = 8            # sine modes of the input family
X = np.linspace(0.0, 1.0, NX)
SCALE = 1.0 / np.arange(1, NMODE + 1)      # std of c_k is 1/k


def grid(nx: int = NX) -> np.ndarray:
    """Uniform grid on [0, 1] with nx points (both boundaries included)."""
    return np.linspace(0.0, 1.0, nx)


def sample_family(n: int, seed: int, nmode: int = NMODE) -> np.ndarray:
    """n coefficient vectors c (shape (n, nmode)) of the sine family, c_k ~ N(0, 1/k^2)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, nmode)) / np.arange(1, nmode + 1)


def profile(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The initial profile a(x) = sum_k c_k sin(k pi x) for coefficients c."""
    c = np.asarray(c)
    return sum(c[k] * np.sin((k + 1) * np.pi * x) for k in range(len(c)))


def profile_xx(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Second derivative a''(x) of the profile (needed by hard-constrained PI-DeepONets)."""
    c = np.asarray(c)
    return -sum(c[k] * ((k + 1) * np.pi) ** 2 * np.sin((k + 1) * np.pi * x) for k in range(len(c)))


def exact(c: np.ndarray, x: np.ndarray, t: float, alpha: float = ALPHA) -> np.ndarray:
    """Analytic solution T(x, t) of the unforced rod for a family member c."""
    c = np.asarray(c)
    return sum(c[k] * np.exp(-alpha * ((k + 1) * np.pi) ** 2 * t) * np.sin((k + 1) * np.pi * x)
               for k in range(len(c)))


def steps(t_end: float, nx: int = NX, alpha: float = ALPHA, cfl: float = 0.4):
    """(n_steps, dt) of the FTCS march: stable (cfl <= 0.5) and landing exactly on t_end."""
    dx = 1.0 / (nx - 1)
    dt_cfl = cfl * dx ** 2 / alpha
    n = max(1, math.ceil(t_end / dt_cfl))
    return n, t_end / n


def ftcs(u0: np.ndarray, t_end: float = T_F, alpha: float = ALPHA, q=None, nx: int | None = None,
         snapshots=None, c: float = 0.0):
    """Explicit FTCS march of an initial profile u0 (given on the grid) to t_end.

    q : optional source q(x) on the grid (array) added each step.
    c : optional advection speed; zero preserves the Part V rod exactly.
        Nonzero c uses upwind advection and a combined explicit CFL bound.
    snapshots : optional list of times; then returns (T_end, [T at each time]).
    """
    u = np.asarray(u0, dtype=float).copy()
    nx = len(u) if nx is None else nx
    dx = 1.0 / (nx - 1)
    n, dt = steps(t_end, nx, alpha)
    if c:
        n=max(n,math.ceil(t_end*(2*alpha/dx**2+abs(c)/dx)/.8))
        dt=t_end/n
    u[0] = u[-1] = 0.0
    want = {} if snapshots is None else {int(round(t / dt)): i for i, t in enumerate(snapshots)}
    out = [None] * len(want)
    if 0 in want:
        out[want[0]] = u.copy()
    src = 0.0 if q is None else np.asarray(q)
    for k in range(1, n + 1):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
        adv = np.zeros_like(u)
        if c > 0: adv[1:-1] = c*(u[1:-1]-u[:-2])/dx
        elif c < 0: adv[1:-1] = c*(u[2:]-u[1:-1])/dx
        u = u + dt * (alpha * lap - adv + src)
        u[0] = u[-1] = 0.0
        if k in want:
            out[want[k]] = u.copy()
    return (u, out) if snapshots is not None else u


def solve_family(C: np.ndarray, t_end: float = T_F, x: np.ndarray = X, alpha: float = ALPHA) -> np.ndarray:
    """FTCS solutions T(x, t_end) for every coefficient row of C; shape (len(C), len(x))."""
    return np.array([ftcs(profile(c, x), t_end, alpha) for c in C])


def dataset(n: int, seed: int, sensors: np.ndarray, t_end: float = T_F, x: np.ndarray = X):
    """Inputs at the sensors (n, m), targets on x (n, len(x)) and the coefficients (n, nmode)."""
    C = sample_family(n, seed)
    U = np.array([profile(c, sensors) for c in C])
    S = solve_family(C, t_end, x)
    return U, S, C


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Relative L2 error per row: ||pred - truth|| / ||truth||."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    return np.linalg.norm(pred - truth, axis=-1) / (np.linalg.norm(truth, axis=-1) + 1e-12)


def check_reference(n: int = 20, seed: int = 123, t_end: float = T_F) -> float:
    """Max relative error of FTCS against the analytic series on n family members."""
    C = sample_family(n, seed)
    return max(float(relative_l2(ftcs(profile(c, X), t_end), exact(c, X, t_end))) for c in C)


if __name__ == "__main__":
    n, dt = steps(T_F)
    print(f"FTCS: {n} steps of dt={dt:.3e}; max relative error vs analytic series {check_reference():.2e}")

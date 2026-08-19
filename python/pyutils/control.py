"""Plumbing shared by the control modules.

Deliberately thin. LQR, MPC and the policy-gradient loop are the *content* of
those lectures, so they are written out in the notebooks where a reader can see
them. What lives here is the boring machinery every one of them needs and none
of them teaches: turning a nonlinear plant into the linear discrete model a
controller wants, and running a closed loop while recording what happened.

    from pyutils.control import linearize, discretize, simulate_closed_loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["linearize", "discretize", "ClosedLoopLog", "simulate_closed_loop"]


def linearize(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    z0: np.ndarray,
    u0: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians of ``f(z, u)`` about an operating point.

    Returns the continuous-time pair ``(A, B)``. Central differences rather
    than forward: the error is O(eps^2), which matters because these matrices
    are then fed to a Riccati solve that squares things.
    """
    z0 = np.asarray(z0, float)
    u0 = np.asarray(u0, float)
    n, m = z0.size, u0.size

    A = np.zeros((n, n))
    for j in range(n):
        dz = np.zeros(n)
        dz[j] = eps
        A[:, j] = (f(z0 + dz, u0) - f(z0 - dz, u0)) / (2 * eps)

    B = np.zeros((n, m))
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        B[:, j] = (f(z0, u0 + du) - f(z0, u0 - du)) / (2 * eps)

    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation via the matrix exponential.

    Uses the block-matrix trick: exp([[A, B], [0, 0]] dt) has Ad in its top-left
    block and Bd in its top-right. Exact for a ZOH input, unlike Euler, which
    quietly biases the identified model at large dt.
    """
    from scipy.linalg import expm

    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


@dataclass
class ClosedLoopLog:
    """What a closed-loop run produced. Arrays are (steps, dim)."""

    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    z: np.ndarray = field(default_factory=lambda: np.empty(0))
    u: np.ndarray = field(default_factory=lambda: np.empty(0))
    solve_ms: np.ndarray = field(default_factory=lambda: np.empty(0))

    def cost(self, Q: np.ndarray, R: np.ndarray, z_ref: np.ndarray | None = None) -> float:
        """Accumulated quadratic stage cost -- the number controllers compete on."""
        e = self.z if z_ref is None else self.z - z_ref
        return float(np.einsum("ti,ij,tj->", e, Q, e) + np.einsum("ti,ij,tj->", self.u, R, self.u))


def simulate_closed_loop(
    step: Callable[[np.ndarray, np.ndarray], np.ndarray],
    policy: Callable[[int, np.ndarray], np.ndarray],
    z0: np.ndarray,
    n_steps: int,
    dt: float,
) -> ClosedLoopLog:
    """Run ``policy`` against ``step`` and record the trajectory.

    ``step(z, u) -> z_next`` is the plant for one sample; ``policy(k, z) -> u``
    is the controller. Solve time is recorded per step because "does it fit in
    the sample interval" is a real-time claim these lectures have to make with
    a number rather than a hope.
    """
    import time

    z = np.asarray(z0, float)
    zs, us, ms = [], [], []
    for k in range(n_steps):
        t0 = time.perf_counter()
        u = np.atleast_1d(np.asarray(policy(k, z), float))
        ms.append((time.perf_counter() - t0) * 1e3)
        zs.append(z.copy())
        us.append(u.copy())
        z = np.asarray(step(z, u), float)

    return ClosedLoopLog(
        t=np.arange(n_steps) * dt,
        z=np.array(zs),
        u=np.array(us),
        solve_ms=np.array(ms),
    )

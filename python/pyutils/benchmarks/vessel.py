"""3-DOF surface-vessel maneuvering model (Fossen form) + RK4 reference solver.

This is the course's ODE benchmark. The same model is re-used by the
data-driven, PINN and CoSTA modules, so the reference solution produced here is
the ground truth those methods are judged against.

Model (body frame, surge-sway-yaw):

    eta_dot = R(psi) nu
    M nu_dot + C(nu) nu + D(nu) nu = tau

with pose  eta = [x, y, psi]  (NED) and body velocity  nu = [u, v, r].

Notation in the course: scalars italic (u, v, r, psi); vectors bold
(eta, nu, tau); matrices bold capital (M, C, D, R).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class VesselParams:
    """Rigid-body + hydrodynamic coefficients of a small surface vessel.

    The rigid-body inertia is 'known'; the added-mass and damping entries are
    the 'uncertain' coefficients later modules identify or correct.
    """

    m: float = 50.0          # mass [kg]
    Iz: float = 20.0         # yaw inertia [kg m^2]
    # added mass (negative by convention) -> uncertain
    X_udot: float = -5.0
    Y_vdot: float = -30.0
    N_rdot: float = -10.0
    # linear damping (positive, dissipative) -> uncertain
    Du: float = 10.0
    Dv: float = 40.0
    Dr: float = 15.0
    # quadratic damping -> uncertain
    Xuu: float = 5.0
    Yvv: float = 40.0
    Nrr: float = 10.0

    def M(self) -> np.ndarray:
        """Mass matrix M = M_RB + M_A (diagonal for this trimmed vessel)."""
        return np.diag([self.m - self.X_udot,
                        self.m - self.Y_vdot,
                        self.Iz - self.N_rdot])

    def C(self, nu: np.ndarray) -> np.ndarray:
        """Coriolis-centripetal matrix C(nu) (rigid-body + added mass)."""
        u, v, _ = nu
        m11 = self.m - self.X_udot
        m22 = self.m - self.Y_vdot
        c13 = -m22 * v
        c23 = m11 * u
        return np.array([[0.0, 0.0, c13],
                         [0.0, 0.0, c23],
                         [-c13, -c23, 0.0]])

    def D(self, nu: np.ndarray) -> np.ndarray:
        """Hydrodynamic damping D(nu) = linear + nonlinear (positive definite)."""
        u, v, r = nu
        return np.diag([self.Du + self.Xuu * abs(u),
                        self.Dv + self.Yvv * abs(v),
                        self.Dr + self.Nrr * abs(r)])


def rotation(psi: float) -> np.ndarray:
    """Rotation matrix R(psi) from body frame to NED."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def state_derivative(z: np.ndarray, tau: np.ndarray, p: VesselParams) -> np.ndarray:
    """Right-hand side f(z, tau) of the 6-D state z = [eta, nu]."""
    eta, nu = z[:3], z[3:]
    eta_dot = rotation(eta[2]) @ nu
    nu_dot = np.linalg.solve(p.M(), tau - p.C(nu) @ nu - p.D(nu) @ nu)
    return np.concatenate([eta_dot, nu_dot])


def linear_stability(p: VesselParams) -> np.ndarray:
    """Eigenvalues of the velocity dynamics linearized at rest.

    Asymptotically stable iff all eigenvalues have negative real part -- the
    ODE analogue of the von Neumann stability test for the PDE schemes.
    """
    A = -np.linalg.solve(p.M(), p.D(np.zeros(3)))
    return np.linalg.eigvals(A)


def simulate(tau_fn: Callable[[float, np.ndarray], np.ndarray],
             z0: np.ndarray | None = None,
             t_end: float = 60.0,
             dt: float = 0.02,
             p: VesselParams | None = None):
    """Integrate the vessel with RK4 under a thruster law tau_fn(t, z).

    Returns (t, Z) with Z of shape (n_steps+1, 6): columns [x, y, psi, u, v, r].
    """
    p = p or VesselParams()
    z = np.zeros(6) if z0 is None else np.asarray(z0, float).copy()
    n = int(round(t_end / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    Z = np.empty((n + 1, 6))
    Z[0] = z
    for k in range(n):
        tk = t[k]
        k1 = state_derivative(z, tau_fn(tk, z), p)
        k2 = state_derivative(z + 0.5 * dt * k1, tau_fn(tk + 0.5 * dt, z), p)
        k3 = state_derivative(z + 0.5 * dt * k2, tau_fn(tk + 0.5 * dt, z), p)
        k4 = state_derivative(z + dt * k3, tau_fn(tk + dt, z), p)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Z[k + 1] = z
    return t, Z


# ----------------------------- standard maneuvers -----------------------------

def turning_circle(surge: float = 200.0, yaw: float = 80.0):
    """Constant surge thrust and yaw moment -> a turning circle."""
    return lambda t, z: np.array([surge, 0.0, yaw])


def zigzag(surge: float = 200.0, yaw: float = 120.0, psi_switch: float = 0.35):
    """20/20-style zig-zag: flip the yaw moment when heading crosses +/-psi_switch."""
    def tau(t, z):
        sign = -1.0 if z[2] > psi_switch else (1.0 if z[2] < -psi_switch else 1.0)
        return np.array([surge, 0.0, sign * yaw])
    return tau


if __name__ == "__main__":
    p = VesselParams()
    print("M =\n", p.M())
    print("stability eigenvalues:", linear_stability(p))
    t, Z = simulate(turning_circle(), t_end=40.0)
    print("final pose [x, y, psi]:", Z[-1, :3])

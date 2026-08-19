"""2-D incompressible Navier-Stokes reference solvers.

Two classic benchmarks:

  * `lid_driven_cavity` -- primitive-variable (u, v, p) projection method with a
    Jacobi pressure-Poisson solve. The canonical incompressible test case.
  * `vortex_merger` -- vorticity-streamfunction form on a periodic box, advanced
    with a pseudo-spectral (FFT) Poisson solve; two like-signed vortices merge.

These are the course's second PDE benchmark and the ground truth for the FNO
(module 26) and reduced-order-modeling (module 19) lectures, which target exactly
the cost of these solves.

Notation: velocity is a vector field (u, v); pressure p and vorticity omega are
scalar fields.
"""
from __future__ import annotations

import numpy as np


# ----------------------------- lid-driven cavity ------------------------------

def lid_driven_cavity(nx: int = 41, ny: int = 41, nt: int = 500,
                      nit: int = 50, Re: float = 100.0, U_lid: float = 1.0,
                      L: float = 1.0, dt: float | None = None):
    """Lid-driven cavity via Chorin projection (homogeneous Dirichlet walls,
    moving top lid u=U_lid).

    Returns (x, y, u, v, p) on an (nx, ny) grid. Density is scaled to 1, so
    nu = U_lid * L / Re.
    """
    dx, dy = L / (nx - 1), L / (ny - 1)
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    nu = U_lid * L / Re
    dt = dt or 0.2 * min(dx, dy) ** 2 / nu        # diffusive-safe step
    rho = 1.0

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))

    def laplacian(f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
            + (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
        )
        return lap

    def ddx(f):
        d = np.zeros_like(f)
        d[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
        return d

    def ddy(f):
        d = np.zeros_like(f)
        d[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dy)
        return d

    for _ in range(nt):
        # --- predictor: momentum without the pressure gradient -> u*, v* ---
        un, vn = u.copy(), v.copy()
        u_star = un + dt * (-(un * ddx(un) + vn * ddy(un)) + nu * laplacian(un))
        v_star = vn + dt * (-(un * ddx(vn) + vn * ddy(vn)) + nu * laplacian(vn))

        # boundary conditions on the intermediate velocity
        for f in (u_star, v_star):
            f[0, :] = 0.0; f[-1, :] = 0.0; f[:, 0] = 0.0
        u_star[:, -1] = U_lid; v_star[:, -1] = 0.0

        # --- pressure Poisson: lap(p) = rho/dt * div(u*) (Jacobi iterations) ---
        rhs = rho / dt * (ddx(u_star) + ddy(v_star))
        for _ in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                ((pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy**2
                 + (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx**2
                 - rhs[1:-1, 1:-1] * dx**2 * dy**2)
                / (2 * (dx**2 + dy**2))
            )
            # Neumann p (dp/dn=0) on walls, reference p=0 at top lid
            p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]; p[:, -1] = 0.0

        # --- corrector: project to a divergence-free velocity ---
        u = u_star - dt / rho * ddx(p)
        v = v_star - dt / rho * ddy(p)

        # re-impose velocity BCs
        u[0, :] = 0.0; u[-1, :] = 0.0; u[:, 0] = 0.0; u[:, -1] = U_lid
        v[0, :] = 0.0; v[-1, :] = 0.0; v[:, 0] = 0.0; v[:, -1] = 0.0

    return x, y, u, v, p


# ------------------------- vortex merger (vorticity) --------------------------

def vortex_merger(n: int = 128, nt: int = 2000, Re: float = 2000.0,
                 L: float = 2 * np.pi, dt: float = 1e-3, sep: float = 0.6,
                 save_every: int = 0):
    """Two like-signed Gaussian vortices on a periodic box, vorticity form.

    d omega/dt + (u.grad) omega = nu lap(omega),  lap(psi) = -omega,
    advanced with RK4 and a pseudo-spectral (FFT) Poisson solve.

    Returns (X, Y, omega) for the final field, or (X, Y, omega, frames, times)
    when save_every > 0.
    """
    x = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    nu = 1.0 / Re

    k = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2_inv = np.zeros_like(K2)
    K2_inv[K2 > 0] = 1.0 / K2[K2 > 0]

    # initial condition: two Gaussian blobs of vorticity
    c = L / 2
    def blob(x0, y0):
        return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * 0.25**2))
    omega = blob(c - sep, c) + blob(c + sep, c)

    def rhs(w):
        w_hat = np.fft.fft2(w)
        psi_hat = w_hat * K2_inv
        u = np.real(np.fft.ifft2(1j * KY * psi_hat))     # u =  d psi/dy
        v = np.real(np.fft.ifft2(-1j * KX * psi_hat))    # v = -d psi/dx
        wx = np.real(np.fft.ifft2(1j * KX * w_hat))
        wy = np.real(np.fft.ifft2(1j * KY * w_hat))
        lap = np.real(np.fft.ifft2(-K2 * w_hat))
        return -(u * wx + v * wy) + nu * lap

    frames, times = [], []
    for step in range(nt):
        k1 = rhs(omega)
        k2 = rhs(omega + 0.5 * dt * k1)
        k3 = rhs(omega + 0.5 * dt * k2)
        k4 = rhs(omega + dt * k3)
        omega = omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if save_every and (step % save_every == 0):
            frames.append(omega.copy())
            times.append((step + 1) * dt)

    if save_every:
        return X, Y, omega, frames, times
    return X, Y, omega


if __name__ == "__main__":
    x, y, u, v, p = lid_driven_cavity(nx=41, ny=41, nt=200, Re=100.0)
    speed = np.sqrt(u**2 + v**2)
    print("cavity max speed:", float(speed.max()))
    X, Y, w = vortex_merger(n=64, nt=200, Re=1000.0)
    print("vortex omega range:", float(w.min()), float(w.max()))

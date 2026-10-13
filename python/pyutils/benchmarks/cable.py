"""Radial heat conduction in a power-cable cross-section (worked example).

The physics-based recipe (derive -> discretize -> solve) applied to a problem
straight out of the course's own running examples: the export cable that carries
an offshore wind farm's power ashore, or the umbilical feeding a subsea asset.
Current ``I`` in the copper conductor dissipates ohmic heat ``I^2 R'`` per metre;
that heat must diffuse out through the XLPE insulation and be carried away by the
surrounding water. If the conductor runs hotter than the insulation can take
(~90 C for XLPE) the cable ages prematurely -- so "how much current may I push?"
is a temperature question, and therefore a heat-conduction question.

Axisymmetric transient conduction, per unit length, in cylindrical coordinates::

    rho cp dT/dt = (1/r) d/dr ( k r dT/dr ) + qdot(r)

with a copper core (r < a, uniform generation), an XLPE annulus (a < r < b, no
generation), symmetry at r = 0, and convection to the water at r = b:

    -k dT/dr |_b = h ( T(b) - T_inf ).

Why implicit.  Copper's diffusivity is ~1e-4 m^2/s, so the explicit (FTCS)
stability limit dt <= dr^2/(2 alpha) is punishing -- sub-millisecond steps for a
process with a *minutes-to-hours* time constant. The cable is the course's first
genuinely **stiff** problem, so we integrate with backward Euler (unconditionally
stable) and take physically sized steps. That contrast -- an explicit scheme that
is correct but unaffordable -- is exactly the pressure that later motivates
reduced-order and surrogate models.

Notation: scalar fields/properties italic (T, k, alpha); geometry a, b in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class Cable:
    # --- geometry [m] ---------------------------------------------------
    a: float = 0.016          # conductor (copper) radius  ~ 800 mm^2 core
    b: float = 0.040          # outer insulation radius
    N: int = 160              # radial cells (node 0 at centre, node N at r=b)
    # --- copper conductor -----------------------------------------------
    k_c: float = 400.0        # conductivity [W/m/K]
    rho_c: float = 8960.0     # density [kg/m^3]
    cp_c: float = 385.0       # specific heat [J/kg/K]
    rho_e: float = 1.72e-8    # electrical resistivity [ohm m] (Cu, ~20 C)
    # --- XLPE insulation ------------------------------------------------
    k_i: float = 0.29         # conductivity [W/m/K]
    rho_i: float = 930.0      # density [kg/m^3]
    cp_i: float = 2200.0      # specific heat [J/kg/K]
    # --- electrical & cooling -------------------------------------------
    I: float = 1000.0         # conductor current [A]  (rated; slides use this value)
    h: float = 20.0           # convective coefficient at r=b [W/m^2/K]
    T_inf: float = 20.0       # ambient (water) temperature [C]
    T_init: float = 20.0      # initial temperature [C]

    # --- derived electrical quantities ----------------------------------
    def R_per_length(self) -> float:
        """DC resistance per unit length of the conductor [ohm/m]."""
        return self.rho_e / (np.pi * self.a**2)

    def power_per_length(self, I: float | None = None) -> float:
        """Ohmic loss per unit length I^2 R' [W/m]."""
        I = self.I if I is None else I
        return I**2 * self.R_per_length()

    def qdot(self, I: float | None = None) -> float:
        """Volumetric heat generation in the conductor [W/m^3]."""
        return self.power_per_length(I) / (np.pi * self.a**2)

    # --- radial grid & per-node material fields -------------------------
    def grid(self):
        return np.linspace(0.0, self.b, self.N + 1)

    def _fields(self):
        """Per-node (k, rho*cp) and per-cell generation power, on the FV grid."""
        r = self.grid()
        dr = r[1] - r[0]
        core = r <= self.a                        # node in the conductor
        k = np.where(core, self.k_c, self.k_i)
        rhocp = np.where(core, self.rho_c * self.cp_c, self.rho_i * self.cp_i)
        # finite-volume cell areas (per unit length): annulus between faces
        rf = (np.arange(self.N + 1) + 0.5) * dr   # outer face of each cell
        rf_in = np.clip((np.arange(self.N + 1) - 0.5) * dr, 0.0, None)
        area = np.pi * (np.minimum(rf, self.b) ** 2 - rf_in**2)
        # generation power per cell: qdot times the part of the cell inside r<=a
        a_out = np.clip(rf, 0.0, self.a)
        a_in = np.clip(rf_in, 0.0, self.a)
        area_cond = np.pi * (a_out**2 - a_in**2)  # conductor area within cell
        Q = self.qdot() * area_cond               # [W/m] generated in each cell
        return r, dr, k, rhocp, area, Q

    def _matrix(self, dt):
        """Assemble the constant tridiagonal backward-Euler operator (banded)."""
        r, dr, k, rhocp, area, _ = self._fields()
        N = self.N
        Cc = rhocp * area                          # heat capacity per cell [J/K/m]
        # face conductances G_{i+1/2} between nodes i and i+1 [W/K/m]
        rface = (np.arange(N) + 0.5) * dr
        khar = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])   # harmonic mean at faces
        G = khar * (2.0 * np.pi * rface) / dr            # length N
        Gconv = self.h * (2.0 * np.pi * self.b)          # convective conductance
        lower = np.zeros(N + 1); upper = np.zeros(N + 1); diag = np.zeros(N + 1)
        for i in range(N + 1):
            gl = G[i - 1] if i > 0 else 0.0
            gu = G[i] if i < N else 0.0
            diag[i] = Cc[i] + dt * (gl + gu)
            if i > 0:
                lower[i] = -dt * gl
            if i < N:
                upper[i] = -dt * gu
        diag[N] += dt * Gconv                            # convection at r=b
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = upper[:-1]                            # super-diagonal
        ab[1, :] = diag                                   # diagonal
        ab[2, :-1] = lower[1:]                            # sub-diagonal
        return ab, Cc, Gconv

    def solve_transient(self, t_end, dt=5.0, save_every=1):
        """Backward-Euler integration to ``t_end`` [s].

        Returns (r, T_final, frames, times) where frames[k] is the radial
        temperature profile at times[k].
        """
        r, dr, k, rhocp, area, Q = self._fields()
        ab, Cc, Gconv = self._matrix(dt)
        n = int(np.ceil(t_end / dt))
        T = np.full(self.N + 1, float(self.T_init))
        frames, times = [T.copy()], [0.0]
        rhs_src = dt * Q.copy()
        rhs_src[-1] += dt * Gconv * self.T_inf
        for step in range(n):
            b = Cc * T + rhs_src
            T = solve_banded((1, 1), ab, b)
            if save_every and (step + 1) % save_every == 0:
                frames.append(T.copy()); times.append((step + 1) * dt)
        return r, T, frames, times

    # --- analytic references (for validation and the ampacity question) --
    def _thermal_coeff(self) -> float:
        """Steady core-over-ambient rise per watt-per-metre, K/(W/m).

        Series of three resistances: conductor self-heating, insulation
        (cylindrical), and outer convection.
        """
        return (
            1.0 / (4.0 * np.pi * self.k_c)                       # conductor core
            + np.log(self.b / self.a) / (2.0 * np.pi * self.k_i)  # insulation
            + 1.0 / (2.0 * np.pi * self.b * self.h)               # convection
        )

    def steady_core_temperature(self, I: float | None = None) -> float:
        """Analytic steady-state conductor-centre temperature [C]."""
        return self.T_inf + self.power_per_length(I) * self._thermal_coeff()

    def ampacity(self, T_limit: float = 90.0) -> float:
        """Current [A] at which the steady core temperature reaches T_limit."""
        p_max = (T_limit - self.T_inf) / self._thermal_coeff()   # [W/m]
        return float(np.sqrt(p_max / self.R_per_length()))


def field_2d(cable: Cable, T_radial: np.ndarray, ngrid: int = 220):
    """Revolve a radial profile into a 2-D cross-section field for plotting.

    Returns (X, Y, T2d) on a square grid of side 2b; points outside the cable
    (r > b) are masked (NaN).
    """
    r = cable.grid()
    xs = np.linspace(-cable.b, cable.b, ngrid)
    X, Y = np.meshgrid(xs, xs)
    rr = np.sqrt(X**2 + Y**2)
    T2d = np.interp(rr, r, T_radial)
    T2d[rr > cable.b] = np.nan
    return X, Y, T2d


if __name__ == "__main__":
    c = Cable()
    print("R' =", c.R_per_length(), "ohm/m ; P' =", c.power_per_length(), "W/m")
    print("steady core T (analytic) =", round(c.steady_core_temperature(), 2), "C")
    print("ampacity (90 C)          =", round(c.ampacity(90.0), 0), "A")
    r, T, frames, times = c.solve_transient(t_end=3 * 3600.0, dt=5.0, save_every=60)
    print("numerical steady core T  =", round(float(T[0]), 2), "C after", times[-1], "s")

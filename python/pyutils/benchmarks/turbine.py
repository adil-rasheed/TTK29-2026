"""Variable-speed variable-pitch wind-turbine (NREL 5-MW-like) rotor ODE.

This is the course's second ODE benchmark, alongside the vessel. A single-mass
drivetrain driven by aerodynamic torque and a baseline controller:

    J omega_dot = T_aero - T_gen,
    T_aero = P_aero / omega,   P_aero = 1/2 rho A Cp(lambda, beta) v^3,
    lambda = omega R / v       (tip-speed ratio),

with the textbook baseline controller:

  * Region II (below rated wind): beta = 0, generator torque T_gen = K omega^2
    with K = 1/2 rho A R^3 Cp_max / lambda_opt^3  (= 1/2 rho pi R^5 Cp_max /
    lambda_opt^3, since A = pi R^2), which tracks lambda_opt (MPPT).
  * Region III (above rated): torque saturates at T_rated and a pitch PI loop
    feathers the blades to hold the rotor at rated speed, shedding excess power.

The aerodynamic map Cp(lambda, beta) is the *uncertain* part (the analogue of the
vessel's hydrodynamic coefficients): later modules learn, correct, or rediscover it.

The physics and default parameters are the same ones used in the module-01
notebook ``wind_turbine.ipynb``; this module packages them so the data-driven,
PINN, operator and CoSTA modules can import one ground truth.

Notation: scalars italic (omega, beta, v, lambda); the state is the scalar rotor
speed omega, integrated in physical time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TurbineParams:
    """Aerodynamic, drivetrain and control parameters of an NREL 5-MW-like turbine.

    The rigid drivetrain inertia is 'known'; the ``Cp`` map (through the
    aerodynamic coefficients baked into :meth:`Cp`) is the 'uncertain' part.
    """

    rho: float = 1.225        # air density [kg/m^3]
    R: float = 63.0           # rotor radius [m]
    J: float = 38.6e6         # drivetrain inertia (low-speed shaft) [kg m^2]
    P_rated: float = 5.0e6    # rated electrical power [W]
    v_in: float = 3.0         # cut-in wind speed [m/s]
    v_out: float = 25.0       # cut-out wind speed [m/s]
    beta_max: float = 40.0    # pitch actuator limit [deg]
    Kp: float = 18.0          # pitch PI proportional gain
    Ki: float = 6.0           # pitch PI integral gain

    @property
    def A(self) -> float:
        """Rotor swept area [m^2]."""
        return np.pi * self.R ** 2

    def Cp(self, lam, beta):
        """Heier empirical power coefficient.

        ``lam`` is the tip-speed ratio and ``beta`` the blade pitch in degrees.
        Vectorized in both arguments; clipped to be non-negative.
        """
        lam = np.maximum(lam, 1e-6)
        li = 1.0 / (1.0 / (lam + 0.08 * beta) - 0.035 / (beta ** 3 + 1.0))
        return np.maximum(
            0.5176 * (116.0 / li - 0.4 * beta - 5.0) * np.exp(-21.0 / li)
            + 0.0068 * lam,
            0.0,
        )

    # --- derived optimum / rated operating point (cached on first access) -----

    def _derived(self):
        cache = getattr(self, "_cache", None)
        if cache is not None:
            return cache
        lg = np.linspace(2.0, 15.0, 5000)
        cp = self.Cp(lg, 0.0)
        lam_opt = float(lg[np.argmax(cp)])
        Cp_max = float(cp.max())
        v_rated = (self.P_rated / (0.5 * self.rho * self.A * Cp_max)) ** (1.0 / 3.0)
        omega_rated = lam_opt * v_rated / self.R
        K = 0.5 * self.rho * self.A * self.R ** 3 * Cp_max / lam_opt ** 3
        T_rated = self.P_rated / omega_rated
        cache = dict(lam_opt=lam_opt, Cp_max=Cp_max, v_rated=v_rated,
                     omega_rated=omega_rated, K=K, T_rated=T_rated)
        object.__setattr__(self, "_cache", cache)
        return cache

    @property
    def lam_opt(self) -> float:
        return self._derived()["lam_opt"]

    @property
    def Cp_max(self) -> float:
        return self._derived()["Cp_max"]

    @property
    def v_rated(self) -> float:
        """Rated wind speed [m/s] (region II / III boundary)."""
        return self._derived()["v_rated"]

    @property
    def omega_rated(self) -> float:
        """Rated rotor speed [rad/s]."""
        return self._derived()["omega_rated"]

    @property
    def K(self) -> float:
        """Region-II optimal torque gain [N m s^2]."""
        return self._derived()["K"]

    @property
    def T_rated(self) -> float:
        """Rated generator torque [N m]."""
        return self._derived()["T_rated"]


def simulate(v_fn: Callable[[float], float],
             p: TurbineParams | None = None,
             t_end: float = 150.0,
             dt: float = 0.05,
             omega0: float | None = None):
    """Integrate the rotor ODE under the baseline controller and a wind law.

    Parameters
    ----------
    v_fn : callable
        Wind speed ``v = v_fn(t)`` [m/s] as a function of time.
    p : TurbineParams, optional
        Turbine parameters (defaults to the NREL 5-MW-like set).
    t_end, dt : float
        Horizon and step of the explicit (semi-implicit) time integration.
    omega0 : float, optional
        Initial rotor speed [rad/s]. Defaults to the region-II tracking guess
        ``min(lam_opt v(0) / R, omega_rated)``.

    Returns
    -------
    t : (n+1,) array of time [s]
    omega : (n+1,) rotor speed [rad/s]
    beta : (n+1,) blade pitch [deg]
    P : (n,) generated electrical power [W]  (one per step)
    Cp : (n,) instantaneous power coefficient
    """
    p = p or TurbineParams()
    R, A, rho, J = p.R, p.A, p.rho, p.J
    K, T_rated = p.K, p.T_rated
    omega_rated, lam_opt = p.omega_rated, p.lam_opt
    Kp, Ki = p.Kp, p.Ki

    n = int(round(t_end / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    if omega0 is None:
        omega0 = min(lam_opt * float(v_fn(0.0)) / R, omega_rated)
    omega = float(omega0)
    beta = 0.0
    integ = 0.0

    OM = np.empty(n + 1); BE = np.empty(n + 1)
    P = np.empty(n); CP = np.empty(n)
    OM[0] = omega; BE[0] = beta
    for k in range(n):
        v = float(v_fn(t[k]))
        cp = float(p.Cp(omega * R / v, beta))
        T_aero = 0.5 * rho * A * cp * v ** 3 / max(omega, 0.1)
        T_gen = min(K * omega ** 2, T_rated)
        # pitch PI to hold rated speed (region III); idle below rated
        err = omega - omega_rated
        integ += err * dt
        beta_cmd = Kp * err + Ki * integ
        beta = min(max(beta_cmd, 0.0), p.beta_max)
        if beta != beta_cmd:                       # anti-windup
            integ = (beta - Kp * err) / Ki
        omega = max(omega + dt * (T_aero - T_gen) / J, 0.1)
        OM[k + 1] = omega; BE[k + 1] = beta
        P[k] = T_gen * omega; CP[k] = cp
    return t, OM, BE, P, CP


def steady_operating_point(v: float,
                           p: TurbineParams | None = None,
                           t_end: float = 150.0,
                           dt: float = 0.05):
    """Steady rotor speed, pitch, power and Cp at a constant wind speed ``v``.

    Integrates to (near) steady state and returns the final values -- the point
    that traces out one sample of the power curve.

    Returns
    -------
    dict with keys ``omega`` [rad/s], ``rpm``, ``beta`` [deg], ``P`` [W], ``Cp``.
    """
    p = p or TurbineParams()
    t, OM, BE, P, CP = simulate(lambda s: v, p=p, t_end=t_end, dt=dt)
    return dict(omega=float(OM[-1]), rpm=float(OM[-1] * 30.0 / np.pi),
                beta=float(BE[-1]), P=float(P[-1]), Cp=float(CP[-1]))


def power_curve(v_grid, p: TurbineParams | None = None,
                t_end: float = 150.0, dt: float = 0.05):
    """Steady power curve over an array of wind speeds.

    Returns
    -------
    P : (len(v_grid),) steady power [W]
    omega : (len(v_grid),) steady rotor speed [rad/s]
    beta : (len(v_grid),) steady pitch [deg]
    """
    p = p or TurbineParams()
    v_grid = np.asarray(v_grid, float)
    P = np.empty_like(v_grid); OM = np.empty_like(v_grid); BE = np.empty_like(v_grid)
    for i, v in enumerate(v_grid):
        op = steady_operating_point(float(v), p=p, t_end=t_end, dt=dt)
        P[i] = op["P"]; OM[i] = op["omega"]; BE[i] = op["beta"]
    return P, OM, BE


def turbulent_wind(v_mean: float, t_end: float = 300.0, dt: float = 0.05,
                   TI: float = 0.12, tau: float = 20.0, seed: int = 1):
    """A low-pass-filtered gust process (Ornstein-Uhlenbeck) around ``v_mean``.

    Returns ``(v_fn, t, v)`` where ``v_fn(s)`` interpolates the sampled series --
    a realistic turbulent wind for driving the closed loop.

    The same OU recursion (u += dt*(-u/tau) + TI*v_mean*sqrt(2 dt/tau)*xi) is
    what the hybrid modules use for the turbine thread; the parameters differ
    by module and are recorded in each results file:
        Module 20 (CoSTA)     TI = 0.08, tau = 25 s, dt = 0.1 s, 250 s windows
        Module 25 (DeepONet)  TI = 0.07, tau = 30 s, dt = 0.1 s, 200 s windows,
                              v_mean in [8, 14] m/s
        Modules 02 and 22 use steady (constant) wind: the power-curve surrogate
        and the one-manoeuvre PINN need no turbulence.
    The wake field of Modules 21, 22 and 26 is a different object (a 2-D
    steady field defined in those modules' notebooks).
    """
    rng = np.random.default_rng(seed)
    n = int(round(t_end / dt))
    sig = TI * v_mean
    u = 0.0
    out = np.empty(n + 1)
    for i in range(n + 1):
        u += dt * (-u / tau) + sig * np.sqrt(2 * dt / tau) * rng.standard_normal()
        out[i] = v_mean + u
    tt = np.linspace(0.0, n * dt, n + 1)
    return (lambda s: float(np.interp(s, tt, out))), tt, out


if __name__ == "__main__":
    p = TurbineParams()
    print(f"lambda_opt={p.lam_opt:.2f}  Cp_max={p.Cp_max:.3f}  "
          f"v_rated={p.v_rated:.1f} m/s  "
          f"omega_rated={p.omega_rated:.3f} rad/s ({p.omega_rated*30/np.pi:.1f} rpm)")
    for v in [5.0, 8.0, 11.0, 15.0, 20.0]:
        op = steady_operating_point(v, p=p)
        print(f"v={v:5.1f} m/s  P={op['P']/1e6:5.2f} MW  "
              f"rpm={op['rpm']:5.1f}  beta={op['beta']:5.1f} deg")

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

How the modules drive it (one place to see the differences; the parameters
above are shared by all of them):

    Module 02  3-DOF sinusoidal forcing, bias 120 N surge, amplitudes 60/30/50
               N, N, N m, 40 s at 0.1 s: a surrogate of the full 3-DOF response.
    Module 20  turning circle and zig-zag manoeuvres with a HIDDEN quadratic drag
               added to the plant, 30 s at 0.02 s: CoSTA learns the drag.
    Module 21  one 80 s excited record for the sparse regression, 40 s test runs.
    Module 22  constant thrust 120 N (and a 15 s turning circle): a PINN solves
               one manoeuvre per training run, so there is no family here.
    Module 25  random_surge_thrust() (surge-only family) under heading_autopilot(),
               20 s at 0.02 s: the force-to-path operator.

The forcing differs because the questions differ (one manoeuvre, a correction,
an equation, an operator); the hull, the coefficients and the RK4 solver do not.
The open-loop course instability is documented at heading_autopilot().
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


def zigzag(surge: float = 200.0, yaw: float = 120.0, psi_switch: float = 0.35,
           hysteresis: bool = False):
    """20/20-style zig-zag: flip the yaw moment when heading crosses +/-psi_switch.

    With ``hysteresis=False`` (the original law, kept so modules 01-02 reproduce)
    the command is memoryless: inside the band |psi| < psi_switch it is always
    +yaw, so the heading oscillates about +psi_switch rather than between the two
    switch lines.  ``hysteresis=True`` is the standard zig-zag: the command keeps
    its previous sign until the heading crosses the *opposite* switch line.  The
    memory is reset whenever the law is called at t <= 0, so one law object can
    drive several runs.
    """
    if not hysteresis:
        def tau(t, z):
            sign = -1.0 if z[2] > psi_switch else (1.0 if z[2] < -psi_switch else 1.0)
            return np.array([surge, 0.0, sign * yaw])
        return tau

    state = {"sign": 1.0}

    def tau(t, z):
        if t <= 0.0:
            state["sign"] = 1.0
        if z[2] > psi_switch:
            state["sign"] = -1.0
        elif z[2] < -psi_switch:
            state["sign"] = 1.0
        return np.array([surge, 0.0, state["sign"] * yaw])
    return tau


if __name__ == "__main__":
    p = VesselParams()
    print("M =\n", p.M())
    print("stability eigenvalues:", linear_stability(p))
    t, Z = simulate(turning_circle(), t_end=40.0)
    print("final pose [x, y, psi]:", Z[-1, :3])


# ---------------------------------------------------------------------------
#  Shared forcing families and the course-stability fact (Modules 02, 20-25)
# ---------------------------------------------------------------------------
def random_surge_thrust(rng, horizon: float = 20.0):
    """The smooth surge-thrust family used by the operator modules: a bias in
    [80, 200] N plus three sinusoids with amplitudes in [-60, 60] N at
    0.3, 0.6 and 0.9 rad/s and random phases.  Returns tau_1(t) as a callable."""
    a = rng.uniform(80.0, 200.0)
    coef = rng.uniform(-60.0, 60.0, 3)
    ph = rng.uniform(0.0, 2.0 * np.pi, 3)
    return lambda t, a=a, coef=coef, ph=ph: a + sum(coef[j] * np.sin((j + 1) * 0.3 * t + ph[j]) for j in range(3))


def heading_autopilot(tau_ext_fn: Callable[[float], np.ndarray], kp: float = 200.0, kd: float = 60.0,
                      psi_ref: float = 0.0) -> Callable[[float, np.ndarray], np.ndarray]:
    """Wrap an external force history tau_ext(t) -> [tau_1, tau_2, tau_3] with a
    PD heading controller on the yaw channel.

    Why this exists: the 3-DOF model above is COURSE-UNSTABLE in open loop at
    speed.  With tau_3 == 0 exactly it sails straight, but any yaw moment, however
    small, grows through the Coriolis coupling and the vessel spins (psi reaches
    about 20 rad in 20 s for a 0.2 N m sinusoidal moment at 100 N surge force).
    A real ASV holds its course with an autopilot; Module 25's force-to-path
    example therefore uses this closed-loop plant.  Modules 20 and 22 force yaw
    for short horizons where the growth stays benign.
    """
    def tau_fn(t, z):
        tau = np.asarray(tau_ext_fn(t), dtype=float).copy()
        tau[2] += kp * (psi_ref - z[2]) - kd * z[5]
        return tau
    return tau_fn


def derivative_batch(z, tau, p=None):
    """Vectorised form of state_derivative, with arbitrary leading batch axes."""
    p = p or VesselParams()
    z, tau = np.asarray(z,float), np.asarray(tau,float)
    psi,u,v,r = (z[...,j] for j in (2,3,4,5))
    c,s = np.cos(psi),np.sin(psi)
    m11,m22,m33 = np.diag(p.M())
    coriolis = np.stack([-m22*v*r,m11*u*r,(m22-m11)*u*v],-1)
    damping = np.stack([(p.Du+p.Xuu*abs(u))*u,(p.Dv+p.Yvv*abs(v))*v,(p.Dr+p.Nrr*abs(r))*r],-1)
    return np.concatenate([np.stack([u*c-v*s,u*s+v*c,r],-1),(tau-coriolis-damping)/[m11,m22,m33]],-1)


def step(z, tau, dt=.5, p=None, current=(0.,0.), substeps=5):
    """RK4 reference at the control sample; current is a force in NED [N]."""
    z=np.asarray(z,float).copy(); tau=np.asarray(tau,float); h=dt/substeps
    def f(zz):
        c,s=np.cos(zz[...,2]),np.sin(zz[...,2])
        env=np.stack([c*current[0]+s*current[1],-s*current[0]+c*current[1],np.zeros_like(c)],-1)
        return derivative_batch(zz,tau+env,p)
    for _ in range(substeps):
        a=f(z); b=f(z+h*a/2); c=f(z+h*b/2); d=f(z+h*c)
        z=z+h*(a+2*b+2*c+d)/6
    return z


def commissioning_record(seed, n_samples=900, amplitude=(8,8,4), dt=.5, noise=2e-4):
    """One dithered open-loop record, no environmental current.

    Returns t (n+1), measured Z (n+1,6), applied TAU (n,3). A clipped AR(1)
    input (rho=.85) excites all three channels. Noise is independent Gaussian
    measurement noise, not process noise. Use commissioning_split(), never
    shuffle transitions: 60% training, 20% validation, 20% test, with a 20-step
    guard at each boundary. No rollout crosses a split or a guard.
    """
    if n_samples < 100 or dt <= 0 or noise < 0:
        raise ValueError('need >=100 samples, positive dt and nonnegative noise')
    rng=np.random.default_rng(seed); amp=np.asarray(amplitude,float)
    U=np.zeros((n_samples,3)); Z=np.zeros((n_samples+1,6)); u=np.zeros(3)
    for k in range(n_samples):
        u=np.clip(.85*u+np.sqrt(1-.85**2)*rng.normal(size=3)*amp/1.6,-amp,amp)
        U[k]=u; Z[k+1]=step(Z[k],u,dt=dt)
    return np.arange(n_samples+1)*dt,Z+rng.normal(0,noise,Z.shape),U


def commissioning_split(n_samples=900, guard=20):
    """Transition indices; every label Z[k+1] stays in its own segment."""
    a,b=int(.6*n_samples),int(.8*n_samples)
    if min(a,b-a,n_samples-b) <= guard:
        raise ValueError('segments shorter than guard')
    return {'train':np.arange(0,a),'validation':np.arange(a+guard,b),
            'test':np.arange(b+guard,n_samples)}

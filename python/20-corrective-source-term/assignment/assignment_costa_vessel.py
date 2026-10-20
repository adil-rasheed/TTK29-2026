# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment — CoSTA vessel: a fouled hull, held-out maneuvers
#
# Solution cells use `### BEGIN/END SOLUTION` (nbgrader). Builds on `notebooks/costa_vessel_ode.ipynb`.
#
# The true vessel has a fouled hull: extra drag the nominal model omits,
# $$\boldsymbol{d}(\boldsymbol{\nu})=\big[-4u,\;-10\,v|v|,\;-3\,r|r|\big]^{\top}.$$
# It depends on the body velocity $\boldsymbol\nu$ only, so a correction $\boldsymbol f_\theta(\boldsymbol\nu)$
# can represent it. You build residual-force labels from training maneuvers, select the network on a
# validation maneuver, and test it on maneuvers it never saw. Position (metres) and heading (degrees) are
# reported separately.

# %%
import sys, os, copy
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import numpy as np, torch
from pyutils import pinn_core as pc
from pyutils.pinn_core import MLP
from pyutils.benchmarks import vessel
torch.set_default_dtype(torch.float64); pc.set_plot_style()

p = vessel.VesselParams(); M = p.M()
T_END, DT = 30.0, 0.02

def nominal_deriv(z, tau):
    return vessel.state_derivative(z, tau, p)

def fouling(nu):
    u, v, r = nu
    return np.array([-4.0*u, -10.0*v*abs(v), -3.0*r*abs(r)])

def true_deriv(z, tau):
    dz = nominal_deriv(z, tau).copy(); dz[3:] += np.linalg.solve(M, fouling(z[3:])); return dz

def rk4(deriv, tau_fn, z0=np.zeros(6)):
    """RK4 from z0; returns times, states and the control actually applied at each step."""
    n = int(round(T_END/DT)); t = np.linspace(0, n*DT, n + 1)
    Z = np.empty((n + 1, 6)); TAU = np.empty((n + 1, 3)); z = np.array(z0, float); Z[0] = z
    for k in range(n):
        tau = tau_fn(t[k], z); TAU[k] = tau
        k1 = deriv(z, tau); k2 = deriv(z + 0.5*DT*k1, tau)
        k3 = deriv(z + 0.5*DT*k2, tau); k4 = deriv(z + DT*k3, tau)
        z = z + DT/6*(k1 + 2*k2 + 2*k3 + k4); Z[k + 1] = z
    TAU[n] = tau_fn(t[n], z)
    return t, Z, TAU

# complete maneuvers, assigned to roles before any sample is built
MANEUVERS = {
    'train': [vessel.turning_circle(120.0, 40.0), vessel.turning_circle(120.0, -40.0),
              vessel.zigzag(100.0, 70.0, 0.4, hysteresis=True)],
    'val':   [vessel.zigzag(110.0, 60.0, 0.5, hysteresis=True)],
    'test':  [vessel.turning_circle(110.0, -30.0), vessel.zigzag(120.0, 80.0, 0.3, hysteresis=True)],
}
DATA = {role: [rk4(true_deriv, f) for f in fs] for role, fs in MANEUVERS.items()}

def position_rmse(Z, Zref):
    return float(np.sqrt(np.mean(np.sum((Z[:, :2] - Zref[:, :2])**2, axis=1))))

def final_heading_error_deg(Z, Zref):
    d = Z[-1, 2] - Zref[-1, 2]
    return float(np.degrees(abs(np.arctan2(np.sin(d), np.cos(d)))))

# %% [markdown]
# ## Task 1 — residual-force labels, trajectory by trajectory
#
# Write `labels(t, Z, TAU)` returning the inputs $\boldsymbol\nu_k$ (shape `(n, 3)`) and the residual force
# $$\boldsymbol f_{\text{ref},k}=\mathbf M\dot{\boldsymbol\nu}_k+\mathbf C(\boldsymbol\nu_k)\boldsymbol\nu_k
# +\mathbf D(\boldsymbol\nu_k)\boldsymbol\nu_k-\boldsymbol\tau_k$$
# for **one** trajectory, with $\boldsymbol\tau_k$ the control that trajectory applied. Estimate
# $\dot{\boldsymbol\nu}$ with `np.gradient` (central differences), separately on each stretch where the control
# is constant: the acceleration jumps when a zig-zag flips its yaw command, and a difference across the jump
# mixes the two sides. Then stack the training
# rows into `F_tr, Y_tr` and the validation rows into `F_va, Y_va`.
#
# The check compares your labels with the hidden drag, which a real log would not let you do. Differentiating
# across a switch leaves label errors of tens of newtons at the switch instants; per-segment differencing
# removes them. For the same reason labels are never differentiated across a trajectory boundary.

# %%
def labels(t, Z, TAU):
    ### BEGIN SOLUTION
    nu = Z[:, 3:]; nu_dot = np.empty_like(nu)
    cuts = np.flatnonzero(np.any(np.diff(TAU, axis=0) != 0, axis=1)) + 1   # first sample of each new segment
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        nu_dot[lo:hi] = np.gradient(nu[lo:hi], t[lo:hi], axis=0, edge_order=2) if hi - lo > 2 else 0.0
    Y = np.array([M @ nu_dot[k] + p.C(nu[k]) @ nu[k] + p.D(nu[k]) @ nu[k] - TAU[k] for k in range(len(t))])
    return nu.copy(), Y
    ### END SOLUTION

F_tr = Y_tr = F_va = Y_va = None
### BEGIN SOLUTION
F_tr, Y_tr = map(np.vstack, zip(*[labels(*tr) for tr in DATA['train']]))
F_va, Y_va = map(np.vstack, zip(*[labels(*tr) for tr in DATA['val']]))
### END SOLUTION
d_tr = np.array([fouling(nu) for nu in F_tr])
dev = np.abs(Y_tr - d_tr).max(1)
print(f'training rows {len(F_tr)}, validation rows {len(F_va)}')
print(f'|label - true drag|: median {np.median(dev):.1e} N, max {dev.max():.1e} N')
assert F_tr.shape[1] == 3 and len(F_tr) == sum(len(tr[0]) for tr in DATA['train'])
assert dev.max() < 1.0, 'labels must match the hidden drag up to finite-difference error; check the switches'

# %% [markdown]
# ## Task 2 — normalization from training rows, and validation-selected training
#
# Compute `Fm, Fs, Ym, Ys` from the **training rows only**. Train `net = MLP([3, 32, 32, 3])` (seed 0, Adam,
# learning rate `2e-3`) on the standardized training rows for `MAX_STEPS` steps. Every `VAL_EVERY` steps,
# roll the corrected model out on the validation maneuver and record its position RMSE; keep and finally
# restore the checkpoint with the lowest value. Store the selected step in `best_step`.

# %%
MAX_STEPS, VAL_EVERY = 2000, 250
torch.manual_seed(0)
net = MLP([3, 32, 32, 3])
Fm = Fs = Ym = Ys = None

def f_theta(nu):
    with torch.no_grad():
        return net(torch.tensor((nu - Fm)/Fs)).numpy()*Ys + Ym

def costa_deriv(z, tau):
    dz = nominal_deriv(z, tau).copy(); dz[3:] += np.linalg.solve(M, f_theta(z[3:])); return dz

best_step, history = None, []
### BEGIN SOLUTION
Fm, Fs = F_tr.mean(0), F_tr.std(0); Ym, Ys = Y_tr.mean(0), Y_tr.std(0)
Ft, Yt = torch.tensor((F_tr - Fm)/Fs), torch.tensor((Y_tr - Ym)/Ys)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
best, best_state = np.inf, None
for k in range(MAX_STEPS + 1):
    if k % VAL_EVERY == 0:
        v = float(np.mean([position_rmse(rk4(costa_deriv, f)[1], Zr)
                           for f, (_, Zr, _) in zip(MANEUVERS['val'], DATA['val'])]))
        history.append((k, v))
        if v < best:
            best, best_step, best_state = v, k, copy.deepcopy(net.state_dict())
    if k == MAX_STEPS:
        break
    opt.zero_grad(); loss = ((net(Ft) - Yt)**2).mean(); loss.backward(); opt.step()
net.load_state_dict(best_state)
### END SOLUTION
print('validation position RMSE by step:', [(k, round(v, 4)) for k, v in history])
print('selected step:', best_step)
assert np.allclose(Fm, F_tr.mean(0)) and np.allclose(Ys, Y_tr.std(0))
assert best_step == history[int(np.argmin([v for _, v in history]))][0]

# %% [markdown]
# ## Task 3 — the held-out test, in physical units
#
# For each test maneuver, simulate the nominal PBM and the frozen CoSTA model from rest and compute, against the
# true trajectory, the position RMSE in metres and the final heading error in degrees (wrapped to
# $[0^\circ,180^\circ]$). Store them in lists `pos_pbm, pos_costa, head_pbm, head_costa` (one entry per test
# maneuver, in the order of `MANEUVERS['test']`).

# %%
pos_pbm, pos_costa, head_pbm, head_costa = [], [], [], []
### BEGIN SOLUTION
for f, (_, Zr, _) in zip(MANEUVERS['test'], DATA['test']):
    Zp = rk4(nominal_deriv, f)[1]; Zc = rk4(costa_deriv, f)[1]
    pos_pbm.append(position_rmse(Zp, Zr)); pos_costa.append(position_rmse(Zc, Zr))
    head_pbm.append(final_heading_error_deg(Zp, Zr)); head_costa.append(final_heading_error_deg(Zc, Zr))
### END SOLUTION
for j in range(len(MANEUVERS['test'])):
    print(f'test maneuver {j}: position RMSE  PBM {pos_pbm[j]:.3f} m  CoSTA {pos_costa[j]:.4f} m   |'
          f'  final heading error  PBM {head_pbm[j]:.1f} deg  CoSTA {head_costa[j]:.2f} deg')
assert len(pos_costa) == len(MANEUVERS['test']) and all(np.isfinite(pos_costa))
assert all(c < 0.25*b for c, b in zip(pos_costa, pos_pbm))

# %% [markdown]
# **Your answer.** Check whether the test maneuvers visit $(u,v,r)$ values outside the range covered by the
# training maneuvers, and say what that implies for how far these test results can be generalized.
#
# ### BEGIN SOLUTION
# Compare per-component minima and maxima of the test velocities with those of `F_tr`. Test states inside the
# training box are interpolation for the network, whatever the maneuver is called; any component outside it is
# extrapolation, where the learned force is unconstrained. The test results therefore support transfer to
# maneuvers whose velocities stay within the training range, not to arbitrary maneuvers.
# ### END SOLUTION

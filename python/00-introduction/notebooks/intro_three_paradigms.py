# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The whole course in one figure
#
# One tiny system — a vessel coasting down in surge,
# $m\,\dot u = -d_1 u - d_2\,u|u|$ — modeled three ways:
#
# 1. **Physics-based (PBM)**: the equation with an imperfect drag
#    coefficient (real models are never exactly right).
# 2. **Data-driven (DDM)**: a neural network fit to a short noisy
#    measurement record, no physics at all.
# 3. **Hybrid**: the imperfect physics *plus* a small learned correction —
#    the course's destination.
#
# We train on $t\in[0,4]$ and ask all three to predict $t\in[4,8]$. The
# resulting figure is the thesis of every lecture that follows.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyutils import set_plot_style, save_dual, fg, muted
from pyutils.pinn_core import MLP, train
set_plot_style()
torch.set_default_dtype(torch.float64)
torch.manual_seed(0); np.random.seed(0)

FIG = os.path.abspath(os.path.join('..', 'figures')); os.makedirs(FIG, exist_ok=True)

# %% [markdown]
# ## 1. Truth, and a short noisy data record

# %%
m_, d1, d2_true = 10.0, 1.0, 2.0
d2_pbm = 1.2                       # the imperfect model's drag (40% off)
u0, T_TRAIN, T_END = 2.5, 4.0, 8.0

def rk4(rhs, u0, dt, t_end):
    n = int(round(t_end/dt)); us = [u0]; u = u0
    for _ in range(n):
        k1 = rhs(u); k2 = rhs(u+dt/2*k1); k3 = rhs(u+dt/2*k2); k4 = rhs(u+dt*k3)
        u = u + dt/6*(k1+2*k2+2*k3+k4); us.append(u)
    return np.linspace(0, t_end, n+1), np.array(us)

rhs_true = lambda u: -(d1*u + d2_true*u*np.abs(u))/m_
rhs_pbm = lambda u: -(d1*u + d2_pbm*u*np.abs(u))/m_

t_ref, u_ref = rk4(rhs_true, u0, 1e-3, T_END)

n_obs, sigma = 30, 0.03
t_obs = np.sort(np.random.uniform(0, T_TRAIN, n_obs))
u_obs = np.interp(t_obs, t_ref, u_ref) + sigma*np.random.randn(n_obs)
print(f"{n_obs} noisy samples on [0, {T_TRAIN}]; we predict out to t={T_END}")

# %% [markdown]
# ## 2. Paradigm 1 — physics only (imperfect)

# %%
_, u_pbm = rk4(rhs_pbm, u0, 1e-3, T_END)
print(f"PBM drag is {d2_pbm} vs true {d2_true}: stable but biased everywhere")

# %% [markdown]
# ## 3. Paradigm 2 — data only
#
# A small network fit to $(t, u)$ pairs. Perfectly reasonable *inside* the
# data — and completely unconstrained beyond it.

# %%
net_ddm = MLP([1, 32, 32, 1])
td = torch.tensor(t_obs.reshape(-1, 1))
ud = torch.tensor(u_obs.reshape(-1, 1))
train(lambda: ((net_ddm(td) - ud)**2).mean(), net_ddm.parameters(),
      steps=3000, lr=2e-3)
with torch.no_grad():
    u_ddm = net_ddm(torch.tensor(t_ref.reshape(-1, 1))).numpy().ravel()

# %% [markdown]
# ## 4. Paradigm 3 — hybrid: imperfect physics + learned correction
#
# Two ideas the course will keep reusing, in miniature:
#
# * **Do not difference noisy data** (it amplifies noise — Modules 20–21):
#   the DDM network is already a smooth interpolant, so differentiate *it*,
#   exactly, by autodiff.
# * **Give the correction structure**: the physics says drag looks like
#   $u|u|$, so fit $g(u) = c\,u|u|$ with one least-squares scalar rather
#   than a free network. Structure is what *extrapolates* — a free network
#   correction nails the data range and falls off Module 2's cliff just
#   outside it (try it!).
#
# Then integrate $\dot u = f_{\text{PBM}}(u) + c\,u|u|$
# (Module 20's CoSTA idea + Module 22's grey-box identification, simplest form).

# %%
# sample the smooth fit inside the data window and differentiate it exactly
t_s = torch.linspace(0.2, T_TRAIN-0.2, 80).reshape(-1, 1).requires_grad_(True)
u_s = net_ddm(t_s)
du_s = torch.autograd.grad(u_s, t_s, torch.ones_like(u_s))[0]
u_s, du_s = u_s.detach(), du_s.detach()
resid = (du_s - (-(d1*u_s + d2_pbm*u_s*torch.abs(u_s))/m_)).numpy().ravel()
feat = (u_s*torch.abs(u_s)).numpy().ravel()

# one-scalar least squares: c = <resid, u|u|> / <u|u|, u|u|>
c_hat = float(feat @ resid / (feat @ feat))
print(f"identified correction  c = {c_hat:+.4f}   "
      f"(truth -(d2_true-d2_pbm)/m = {-(d2_true-d2_pbm)/m_:+.4f})")

def rhs_hybrid(u):
    return rhs_pbm(u) + c_hat*u*abs(u)

_, u_hyb = rk4(rhs_hybrid, u0, 1e-2, T_END)
t_hyb = np.linspace(0, T_END, len(u_hyb))

# %% [markdown]
# ## 5. The verdict

# %%
mask_ex = t_ref >= T_TRAIN
u_hyb_i = np.interp(t_ref, t_hyb, u_hyb)

def err(u, region):
    return np.sqrt(np.mean((u[region]-u_ref[region])**2))

rows = [("PBM (wrong drag)", u_pbm), ("DDM (network on data)", u_ddm),
        ("Hybrid (PBM + correction)", u_hyb_i)]
print(f"{'model':28s} {'in-dist RMSE':>14s} {'extrapolation RMSE':>20s}")
for name, u in rows:
    print(f"{name:28s} {err(u, ~mask_ex):14.4f} {err(u, mask_ex):20.4f}")

e_pbm, e_ddm, e_hyb = (err(u, mask_ex) for _, u in rows)
assert e_hyb < e_pbm/3, "the hybrid should beat its own imperfect physics"
assert e_hyb < e_ddm/3, "the hybrid should beat the pure network out of sample"

def plot_intro_three_paradigms():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axvspan(T_TRAIN, T_END, color=muted(), alpha=0.12)
    ax.text(0.98, 0.96, 'extrapolation', transform=ax.transAxes, ha='right',
            va='top', fontsize=10, color=muted())
    ax.plot(t_ref, u_ref, color=fg(), ls='--', lw=1.6, label='truth')
    ax.plot(t_obs, u_obs, 'o', color='C5', ms=4.5, label='noisy data (training)')
    ax.plot(t_ref, u_pbm, color='C0', lw=1.9,
            label=f'physics only, wrong drag  (RMSE {e_pbm:.3f})')
    ax.plot(t_ref, u_ddm, color='C3', lw=1.9,
            label=f'data only, NN  (RMSE {e_ddm:.3f})')
    ax.plot(t_ref, u_hyb_i, color='C2', lw=2.3,
            label=f'hybrid: physics + learned correction  (RMSE {e_hyb:.3f})')
    ax.axvline(T_TRAIN, color=fg(), lw=0.8, ls=':')
    ax.set(xlabel='$t$ [s]', ylabel='surge speed $u(t)$ [m/s]',
           title='One system, three modeling paradigms (RMSE on the shaded region)')
    ax.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(1.0, 0.90))

save_dual(plot_intro_three_paradigms, 'intro_three_paradigms', FIG)
plt.show()

# %% [markdown]
# ## What this course does with that figure
#
# * The blue curve is Modules 1 (build the physics) and its honest limits.
# * The red curve is Module 2 — powerful, and the extrapolation cliff is a
#   theorem-grade warning, not an implementation bug.
# * The green curve is where the course lives: physics carrying the
#   structure, data fixing what the physics gets wrong — via the loss
#   (Module 22), via operators (25–26), via corrections (20), via discovered
#   equations (21), compressed to run in real time (5, 19).
# * Every number above is asserted; every figure in the course is generated
#   by a notebook like this one.

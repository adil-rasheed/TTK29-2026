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
# # Assignment — CoSTA heat: labels, splits, validation and a held-out test
#
# Solution cells use `### BEGIN/END SOLUTION` (nbgrader). Builds on `notebooks/costa_heat_pde.ipynb`.
# The implicit solver, the manufactured reference and the rollout are provided. You build the labels, the
# split, a validation-selected model and its held-out test, following the protocol of the lecture.

# %%
import sys, os, copy
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import numpy as np, torch
from pyutils import pinn_core as pc
from pyutils.pinn_core import MLP
torch.set_default_dtype(torch.float64); pc.set_plot_style()

D = 0.5; N = 41; T_end = 1.0; nt = 50
x = np.linspace(0, 1, N); dx = x[1] - x[0]; dt = T_end / nt
r = D*dt/dx**2; I = N - 2
A = (1 + 2*r)*np.eye(I) - r*np.eye(I, k=1) - r*np.eye(I, k=-1); Ainv = np.linalg.inv(A)

def step(Tn, src):
    """One backward-Euler PBM step with interior source `src`; Dirichlet T = 1 at both ends."""
    b = Tn[1:-1] + dt*np.asarray(src, float); b = b.copy(); b[0] += r; b[-1] += r
    T = Tn.copy(); T[1:-1] = Ainv @ b; T[0] = T[-1] = 1.0
    return T

def Tstar(a, t):
    return 1 + a*np.sin(np.pi*x)*np.exp(-t)

def reference(a):
    return np.array([Tstar(a, n*dt) for n in range(nt + 1)])

def rollout(a, corr=None):
    """Recursive simulation from the exact initial field; corr(T) returns the interior correction."""
    T = Tstar(a, 0.0); out = [T]
    for n in range(nt):
        T = step(T, np.zeros(I) if corr is None else corr(T)); out.append(T)
    return np.array(out)

def rmse(P, R):
    return float(np.sqrt(np.mean((P - R)**2)))

# %% [markdown]
# ## Task 1 — the exact discrete label
#
# Write `label(Tn, Tn1)` returning the exact discrete corrective source at the interior nodes for the step
# from `Tn` to `Tn1`: the residual of the backward-Euler step evaluated on two consecutive reference states,
# $$\sigma^{*,n}_i=\frac{T^{n+1}_i-T^{n}_i}{\Delta t}-D\,\frac{T^{n+1}_{i+1}-2T^{n+1}_i+T^{n+1}_{i-1}}{\Delta x^2}.$$
# The checks verify that inserting it into one solver step reproduces the next reference state, and that a
# zero correction reproduces the PBM exactly.

# %%
def label(Tn, Tn1):
    ### BEGIN SOLUTION
    lap = (Tn1[:-2] - 2*Tn1[1:-1] + Tn1[2:]) / dx**2
    return (Tn1[1:-1] - Tn[1:-1])/dt - D*lap
    ### END SOLUTION

R = reference(2.0)
worst = max(np.abs(step(R[n], label(R[n], R[n+1])) - R[n+1]).max() for n in range(nt))
print('exact label, one step from the reference: max error', worst)
assert worst < 1e-10
assert np.array_equal(rollout(2.0, lambda T: np.zeros(I)), rollout(2.0))

# %% [markdown]
# ## Task 2 — split whole trajectories, then build rows
#
# Use the amplitudes below. Build the feature rows $[T^n_{i-1},T^n_i,T^n_{i+1}]$ (time level $n$ only) and
# labels for the **training** and **validation** trajectories, then compute the normalization statistics
# `Fm, Fs, Ym, Ys` (means and standard deviations of features and labels) from the **training rows only**.

# %%
A_TRAIN = [0.5, 1.0, 1.5, 2.5, 3.0]
A_VAL = [0.75, 2.0]
A_TEST = {'interpolation': [1.25, 2.75], 'extrapolation': [3.5]}

def rows(amps):
    F, Y = [], []
    for a in amps:
        R = reference(a)
        for n in range(nt):
            F.append(np.stack([R[n][:-2], R[n][1:-1], R[n][2:]], 1)); Y.append(label(R[n], R[n+1]))
    return np.vstack(F), np.concatenate(Y)[:, None]

F_tr = Y_tr = F_va = Y_va = None
Fm = Fs = Ym = Ys = None
### BEGIN SOLUTION
F_tr, Y_tr = rows(A_TRAIN); F_va, Y_va = rows(A_VAL)
Fm, Fs = F_tr.mean(0), F_tr.std(0); Ym, Ys = Y_tr.mean(), Y_tr.std()
### END SOLUTION
test_amps = A_TEST['interpolation'] + A_TEST['extrapolation']
assert not (set(A_TRAIN) & set(A_VAL)) and not (set(A_TRAIN + A_VAL) & set(test_amps)), 'roles must be disjoint'
assert F_tr.shape == (len(A_TRAIN)*nt*I, 3) and F_va.shape == (len(A_VAL)*nt*I, 3)
assert np.allclose(Fm, F_tr.mean(0)) and np.isclose(Ys, Y_tr.std())
print('training rows', F_tr.shape[0], ' validation rows', F_va.shape[0])

# %% [markdown]
# ## Task 3 — train, and select the checkpoint by validation rollouts
#
# Train `net = MLP([3, 32, 32, 1])` (seed 0) with Adam (learning rate `2e-3`) on the standardized training rows
# for `MAX_STEPS` steps. Every `VAL_EVERY` steps, without updating the weights, compute `val_rollout`: the
# mean over the validation amplitudes of the rollout RMSE of `rollout(a, corr)` against `reference(a)`. Keep a
# copy of the weights with the lowest `val_rollout` and restore it at the end. Record the selected step in
# `best_step`.

# %%
MAX_STEPS, VAL_EVERY = 1500, 100
torch.manual_seed(0)
net = MLP([3, 32, 32, 1])

def corr(T):
    """Learned correction at the interior nodes, using the training-set normalization."""
    q = np.stack([T[:-2], T[1:-1], T[2:]], 1)
    with torch.no_grad():
        return net(torch.tensor((q - Fm)/Fs)).numpy().ravel()*Ys + Ym

best_step, history = None, []
### BEGIN SOLUTION
Ft, Yt = torch.tensor((F_tr - Fm)/Fs), torch.tensor((Y_tr - Ym)/Ys)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
best, best_state = np.inf, None
for k in range(MAX_STEPS + 1):
    if k % VAL_EVERY == 0:
        v = float(np.mean([rmse(rollout(a, corr), reference(a)) for a in A_VAL]))
        history.append((k, v))
        if v < best:
            best, best_step, best_state = v, k, copy.deepcopy(net.state_dict())
    if k == MAX_STEPS:
        break
    opt.zero_grad(); loss = ((net(Ft) - Yt)**2).mean(); loss.backward(); opt.step()
net.load_state_dict(best_state)
### END SOLUTION
vals = [v for _, v in history]
print(f'selected checkpoint: step {best_step}, validation rollout RMSE {min(vals):.2e}')
assert len(history) >= MAX_STEPS // VAL_EVERY and best_step == history[int(np.argmin(vals))][0]
assert abs(np.mean([rmse(rollout(a, corr), reference(a)) for a in A_VAL]) - min(vals)) < 1e-12, \
    'the restored network must be the selected checkpoint'

# %% [markdown]
# ## Task 4 — the held-out test
#
# With the network frozen, compute the rollout RMSE of the PBM and of CoSTA for every test amplitude and store
# them in the dictionaries `test_pbm[a]` and `test_costa[a]`. Then compute `T_max_train`, the largest temperature
# in any training trajectory, and `T_max_test[a]`, the largest temperature in each test trajectory. Use these to
# answer, in the markdown cell below: for which test case is the network extrapolating **in its input space**?

# %%
test_pbm, test_costa, T_max_test, T_max_train = {}, {}, {}, None
### BEGIN SOLUTION
for a in test_amps:
    test_pbm[a] = rmse(rollout(a), reference(a)); test_costa[a] = rmse(rollout(a, corr), reference(a))
    T_max_test[a] = float(reference(a).max())
T_max_train = float(max(reference(a).max() for a in A_TRAIN))
### END SOLUTION
for a in test_amps:
    print(f'a = {a:4.2f}: PBM {test_pbm[a]:.2e}   CoSTA {test_costa[a]:.2e}   max T {T_max_test[a]:.2f}'
          f'   (training max {T_max_train:.2f})')
assert all(np.isfinite(test_costa[a]) for a in test_amps)
assert all(test_costa[a] < 0.05*test_pbm[a] for a in A_TEST['interpolation'])

# %% [markdown]
# **Your answer.** Which test case requires the network to extrapolate in its input space, and how does that
# show up in the errors?
#
# ### BEGIN SOLUTION
# Only $a=3.5$: its temperatures reach $4.5$, above the training maximum of $4.0$, so the network is queried
# outside the range it was fitted on. Its CoSTA error is clearly larger than for the two interpolation cases,
# though still far below the PBM's. The interpolation amplitudes stay inside the visited temperature range.
# ### END SOLUTION

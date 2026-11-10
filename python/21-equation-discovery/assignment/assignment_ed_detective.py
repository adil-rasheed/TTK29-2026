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
# # Assignment — the discovery detective
#
# Sparse regression always returns *an* equation. In this assignment you get three data sets, each with a hidden defect
# from the lecture's section *When discovery fails*. For each one you (a) **diagnose** the failure with a number you
# compute, and (b) **repair** it.
#
# | Case | System | Hidden defect | Your diagnostic | Your repair |
# |---|---|---|---|---|
# | 1 | surface vessel, sway equation | a term missing from the library | error floor; residual correlation | find the term |
# | 2 | a feedback-controlled system | input collinear with a library column | condition number | excite the input |
# | 3 | a forced, damped oscillator | a noisy (differentiated) regressor | SNR and the attenuation formula | weak form |
#
# Solution cells use `### BEGIN SOLUTION` / `### END SOLUTION` (nbgrader); every task ends with `assert`s that grade it.
# Use the helper `stlsq` below (columns *and* target normalized, so the threshold $\lambda$ is a dimensionless fraction).

# %%
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import numpy as np, matplotlib.pyplot as plt
from pyutils import set_plot_style; set_plot_style()
from pyutils.benchmarks import vessel
trapz = getattr(np, 'trapezoid', None) or np.trapz      # numpy 2.x / 1.x

def stlsq(Theta, y, lam, n_iter=20):
    # sequentially thresholded least squares on normalized columns and target (lam is dimensionless)
    norm = np.linalg.norm(Theta, axis=0) + 1e-12; Tn = Theta/norm; ny = np.linalg.norm(y); yn = y/ny
    xi = np.linalg.lstsq(Tn, yn, rcond=None)[0]
    for _ in range(n_iter):
        small = np.abs(xi) < lam; xi[small] = 0; big = ~small
        if big.any(): xi[big] = np.linalg.lstsq(Tn[:, big], yn, rcond=None)[0]
    return xi*ny/norm

def rel_err(Theta, xi, y): return float(np.linalg.norm(Theta @ xi - y)/np.linalg.norm(y))
LAMS = np.logspace(-3, 0, 13)                            # threshold sweep used throughout

# %% [markdown]
# ## Case 1 — a library with a hole
#
# The vessel of the lecture, $80\dot v=\tau_2-55\,ur-40\,v-40\,|v|v$ for the sway equation. The data below are clean
# (exact derivatives). Your library was written by a colleague who forgot one term.

# %%
p = vessel.VesselParams()
tau_fn = lambda t, z: np.array([150 + 100*np.sin(0.3*t), 90*np.sin(0.5*t + 1), 60*np.sin(0.7*t)])
t, Z = vessel.simulate(tau_fn, t_end=80.0, dt=0.02, p=p)
nu = Z[:, 3:]; tau = np.array([tau_fn(tk, zk) for tk, zk in zip(t, Z)])
nudot = np.array([vessel.state_derivative(z, tau_fn(tk, z), p)[3:] for tk, z in zip(t, Z)])
u, v, r = nu.T; vdot = nudot[:, 1]
LIB = {'1': np.ones_like(u), 'u': u, 'v': v, 'r': r, '|u|u': np.abs(u)*u, '|r|r': np.abs(r)*r,
       'uv': u*v, 'ur': u*r, 'vr': v*r, 'tau1': tau[:, 0], 'tau2': tau[:, 1], 'tau3': tau[:, 2]}
names = list(LIB); Theta = np.stack([LIB[n] for n in names], 1)
print('library:', names)

# %% [markdown]
# **1a.** Sweep the threshold over `LAMS`, fit the $\dot v$ equation at each, and store the smallest relative fit error in
# `floor`. With a complete library and clean data it would be at rounding level ($\sim10^{-15}$).
#
# **1b.** Compute the residual of the *dense* least-squares fit, $\dot v-\boldsymbol\Theta\hat{\boldsymbol\xi}$, and its correlation
# with each candidate in `CANDS`. Store the correlations in the dict `corr`.

# %%
CANDS = {'v^3': v**3, '|v|v': np.abs(v)*v, 'v r^2': v*r**2, 'u^2 v': u**2*v, 'v^2': v**2}
floor = None; corr = None
### BEGIN SOLUTION
floor = min(rel_err(Theta, stlsq(Theta, vdot, lam), vdot) for lam in LAMS)
xi_dense = np.linalg.lstsq(Theta, vdot, rcond=None)[0]; res = vdot - Theta @ xi_dense
corr = {k: float(np.corrcoef(res, c)[0, 1]) for k, c in CANDS.items()}
### END SOLUTION
print(f'error floor = {floor:.3f}'); print({k: round(c, 3) for k, c in corr.items()})
assert floor > 0.1, 'the incomplete library should leave a large error floor'
assert set(corr) == set(CANDS)

# %% [markdown]
# **1c.** The residual points to an *odd* function of $v$ --- but which? Correlation alone is ambiguous here. For each of the
# two best-correlated candidates, add it to the library, redo the sweep, and record the new floor in `floor_with`
# (a dict). Keep the candidate that removes the floor, refit with $\lambda=0.05$, and store its coefficient in `xi_new`
# (true value $-40/80=-0.5$).

# %%
floor_with = {}; best_term = None; xi_new = None
### BEGIN SOLUTION
top2 = sorted(corr, key=lambda k: -abs(corr[k]))[:2]
for k in top2:
    T2 = np.column_stack([Theta, CANDS[k]])
    floor_with[k] = min(rel_err(T2, stlsq(T2, vdot, lam), vdot) for lam in LAMS)
best_term = min(floor_with, key=floor_with.get)
T2 = np.column_stack([Theta, CANDS[best_term]]); xi2 = stlsq(T2, vdot, 0.05); xi_new = float(xi2[-1])
### END SOLUTION
print({k: f'{f:.2e}' for k, f in floor_with.items()}, '->', best_term, f'coefficient {xi_new:+.4f}')
assert best_term == '|v|v' and floor_with['|v|v'] < 1e-8 and abs(xi_new + 0.5) < 1e-6

# %% [markdown]
# *Question (answer in a markdown cell):* $v^3$ correlated more strongly with the residual than $|v|v$, yet it was the wrong
# term. Why can the residual's correlation mislead, and why is the error floor the decisive test?

# %% [markdown]
# ## Case 2 — a controller that hides the dynamics
#
# The plant is $\dot x=-a\,x-b\,x^3+g\,u$ with unknown $a,b,g$. It runs under feedback $u=-k\,x^3$ (closed loop). The
# library is $\{x,\ x^3,\ u\}$.

# %%
A_TRUE, B_TRUE, G_TRUE, K = 1.0, 0.5, 2.0, 1.0
def simulate_plant(dither, t_end=60.0, dt=0.01, x0=1.5):
    n = int(t_end/dt); tt = np.arange(n)*dt; X = np.empty(n); U = np.empty(n); XD = np.empty(n); x = x0
    for i in range(n):
        uc = -K*x**3 + dither(tt[i])
        d = -A_TRUE*x - B_TRUE*x**3 + G_TRUE*uc
        X[i], U[i], XD[i] = x, uc, d; x = x + dt*d
    return tt, X, U, XD
tt, X, U, XD = simulate_plant(lambda s: 0.0)                       # closed loop, no excitation

# %% [markdown]
# **2a.** Build $\boldsymbol\Theta=[x,\ x^3,\ u]$, normalize its columns to unit length, and store the condition number in
# `cond_closed`. Fit by least squares and store the estimates in `est_closed = (a, b, g)`.
#
# **2b.** Repeat with an *excited* input: add the dither $0.8\sin0.5t+0.5\sin1.3t$ to the controller. Store `cond_excited`
# and `est_excited`.

# %%
cond_closed = est_closed = cond_excited = est_excited = None
### BEGIN SOLUTION
def identify(X, U, XD):
    Th = np.column_stack([X, X**3, U]); cond = np.linalg.cond(Th/np.linalg.norm(Th, axis=0))
    c = np.linalg.lstsq(Th, XD, rcond=None)[0]
    return float(cond), (-c[0], -c[1], c[2])
cond_closed, est_closed = identify(X, U, XD)
_, X2, U2, XD2 = simulate_plant(lambda s: 0.8*np.sin(0.5*s) + 0.5*np.sin(1.3*s))
cond_excited, est_excited = identify(X2, U2, XD2)
### END SOLUTION
print(f'closed loop : cond {cond_closed:.1e}, (a, b, g) = {np.round(est_closed, 3)}')
print(f'excited     : cond {cond_excited:.1e}, (a, b, g) = {np.round(est_excited, 3)}   truth ({A_TRUE}, {B_TRUE}, {G_TRUE})')
assert cond_closed > 1e8 and cond_excited < 100
assert np.allclose(est_excited, (A_TRUE, B_TRUE, G_TRUE), atol=1e-3)

# %% [markdown]
# *Question:* show on paper why $b$ and $g$ cannot be separated in closed loop (substitute $u=-kx^3$). The closed-loop model
# nevertheless predicts the closed-loop data perfectly --- why, and when would it fail?

# %% [markdown]
# ## Case 3 — a noisy regressor shrinks its coefficient
#
# A forced, damped oscillator $\ddot x=-\omega^2x-c\,\dot x+f(t)$, with $\omega^2=4$, $c=0.3$, known forcing
# $f(t)=\sin1.1t+0.5\sin2.7t$. An accelerometer measures $\ddot x$ (small noise); a position sensor measures $x$ with noise;
# the velocity must be computed from the position.

# %%
W2, C_TRUE, DT = 4.0, 0.3, 0.01
f = lambda s: np.sin(1.1*s) + 0.5*np.sin(2.7*s)
def oscillator(t_end=100.0):
    n = int(t_end/DT); tt = np.arange(n)*DT; S = np.zeros((n, 2)); s = np.array([1.0, 0.0])
    F = lambda s, t: np.array([s[1], -W2*s[0] - C_TRUE*s[1] + f(t)])
    for i in range(n):
        S[i] = s; h = DT; k1 = F(s, tt[i]); k2 = F(s + h/2*k1, tt[i] + h/2); k3 = F(s + h/2*k2, tt[i] + h/2)
        k4 = F(s + h*k3, tt[i] + h); s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return tt, S[:, 0], S[:, 1]
tt, Xc, Vc = oscillator(); Ac = -W2*Xc - C_TRUE*Vc + f(tt)
rng = np.random.default_rng(0)
NOISE = 0.02                                                          # position noise, fraction of the signal's std
y = Xc + NOISE*Xc.std()*rng.standard_normal(len(tt))                  # measured position
acc = Ac + 0.01*Ac.std()*rng.standard_normal(len(tt))                 # measured acceleration

# %% [markdown]
# **3a.** Estimate the velocity by `np.gradient(y, DT)`, regress `acc` on $[y,\ \dot y,\ f]$, and store the damping estimate in
# `c_fd`.
#
# **3b.** Compute the SNR of your velocity column against the true velocity `Vc`, $\mathrm{SNR}=\|v\|/\|\dot y-v\|$ (centre $v$),
# and the attenuation prediction $c\,\mathrm{SNR}^2/(1+\mathrm{SNR}^2)$. Store them in `snr` and `c_pred`.
#
# **3c.** Remove the bias with the **weak form**, using the position only. Integrating by parts twice and once,
# $\int\ddot x\,\varphi=\int x\,\ddot\varphi$ and $\int\dot x\,\varphi=-\int x\,\dot\varphi$ for a test function that vanishes (with
# its derivative) at the window ends, so on each window
# $$\int x\,\ddot\varphi\,dt=-\omega^2\!\int x\,\varphi\,dt+c\int x\,\dot\varphi\,dt+\int f\,\varphi\,dt .$$
# Use windows of $600$ samples every $150$ samples and $\varphi=((t-a)(b-t))^4$ (hint: `np.poly1d` gives exact
# derivatives). Store the damping estimate in `c_weak`.

# %%
c_fd = snr = c_pred = c_weak = None
### BEGIN SOLUTION
yd = np.gradient(y, DT)
co = np.linalg.lstsq(np.column_stack([y, yd, f(tt)]), acc, rcond=None)[0]; c_fd = float(-co[1])
snr = float(np.linalg.norm(Vc - Vc.mean())/np.linalg.norm(yd - Vc)); c_pred = C_TRUE*snr**2/(1 + snr**2)
G, b = [], []
for s0 in range(0, len(tt) - 600, 150):
    tk = tt[s0:s0 + 600]; P = np.poly1d([-1, tk[0] + tk[-1], -tk[0]*tk[-1]])**4
    phi, dphi, ddphi = P(tk), P.deriv(1)(tk), P.deriv(2)(tk); yy = y[s0:s0 + 600]
    b.append(trapz(yy*ddphi, tk)); G.append([trapz(yy*phi, tk), trapz(yy*dphi, tk), trapz(f(tk)*phi, tk)])
cw = np.linalg.lstsq(np.array(G), np.array(b), rcond=None)[0]; c_weak = float(cw[1])
### END SOLUTION
print(f'finite differences: c = {c_fd:.4f}   (SNR of the velocity column {snr:.2f}; predicted {c_pred:.4f})')
print(f'weak form         : c = {c_weak:.4f}   (truth {C_TRUE})')
assert abs(c_fd - c_pred) < 0.01 and c_fd < 0.8*C_TRUE
assert abs(c_weak - C_TRUE) < 0.01

# %% [markdown]
# *Question:* repeat 3a–3c with `NOISE = 0.05`. How do `c_fd`, `c_pred` and `c_weak` change, and why does collecting more data
# (a longer record) not fix finite differences?

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
# # Assignment: compressed sensing, from soft-thresholding to sensor placement
#
# Four tasks mirroring the lecture: the proximal operator and ISTA, the
# momentum that turns ISTA into FISTA, recovery of a signal sampled far below
# its Nyquist rate, and QR-pivot sensor placement on a learned POD basis.
# Fill in every `### BEGIN SOLUTION` block; the assert cells grade you.

# %%
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import qr

rng = np.random.default_rng(22)

# %% [markdown]
# ## Task 1 — soft-thresholding and ISTA (3 points)
#
# The LASSO is
# $F(\boldsymbol{s})=\tfrac12\|\boldsymbol{y}-\boldsymbol{\Phi}\boldsymbol{s}\|_2^2
#  +\lambda\|\boldsymbol{s}\|_1$. Its smooth part has gradient
# $\nabla f(\boldsymbol{s})=\boldsymbol{\Phi}^\top(\boldsymbol{\Phi}\boldsymbol{s}-\boldsymbol{y})$,
# Lipschitz with constant $L=\|\boldsymbol{\Phi}\|_2^2$, and the proximal
# operator of $\alpha\|\cdot\|_1$ is the soft-threshold
# $\operatorname{sign}(z)\max(|z|-\alpha,0)$.
#
# Implement:
#
# 1. `soft_threshold(z, a)` — elementwise, vectorised, exact zeros below `a`.
# 2. `lasso_objective(Phi, y, s, lam)` — the value of $F$.
# 3. `ista(Phi, y, lam, n_iter, L)` — returns `(s, hist)` where `hist[k]` is
#    the objective after iteration `k`. Start from $\boldsymbol{s}^0=\mathbf{0}$
#    and use the step
#    $\boldsymbol{s}^{k+1}=\operatorname{SoftThresh}(\boldsymbol{s}^k-\tfrac1L\nabla f(\boldsymbol{s}^k),\ \lambda/L)$.

# %%
### BEGIN SOLUTION
def soft_threshold(z, a):
    """prox of a*||.||_1 : sign(z) * max(|z| - a, 0)."""
    return np.sign(z)*np.maximum(np.abs(z) - a, 0.0)


def lasso_objective(Phi, y, s, lam):
    r = y - Phi @ s
    return 0.5*(r @ r) + lam*np.sum(np.abs(s))


def ista(Phi, y, lam, n_iter, L):
    s = np.zeros(Phi.shape[1])
    hist = np.empty(n_iter)
    for k in range(n_iter):
        s = soft_threshold(s - (Phi.T @ (Phi @ s - y))/L, lam/L)
        hist[k] = lasso_objective(Phi, y, s, lam)
    return s, hist
### END SOLUTION

# %%
# the test problem (do not modify)
N, K, M = 512, 15, 160
support = np.sort(rng.choice(N, K, replace=False))
s_true = np.zeros(N)
s_true[support] = rng.choice([-1.0, 1.0], K)*rng.uniform(1.0, 3.0, K)
Phi = rng.standard_normal((M, N))/np.sqrt(M)
y = Phi @ s_true
L = np.linalg.norm(Phi, 2)**2
LAM = 0.01*np.max(np.abs(Phi.T @ y))

# the prox is characterised pointwise
assert np.allclose(soft_threshold(np.array([-3.0, -0.4, 0.0, 0.4, 3.0]), 1.0),
                   [-2.0, 0.0, 0.0, 0.0, 2.0]), "soft_threshold is wrong"
assert soft_threshold(np.array([0.999]), 1.0)[0] == 0.0, \
    "anything below the threshold must become EXACTLY zero"

# the reference optimum, from an independent solver
from sklearn.linear_model import Lasso
sk = Lasso(alpha=LAM/M, fit_intercept=False, max_iter=200_000, tol=1e-14).fit(Phi, y)
F_STAR = lasso_objective(Phi, y, sk.coef_, LAM)

s_ista, hist_ista = ista(Phi, y, LAM, 4000, L)
F_ista = lasso_objective(Phi, y, s_ista, LAM)
assert np.all(np.diff(hist_ista) <= 1e-12), \
    "the proximal-gradient step never increases the objective"
assert (F_ista - F_STAR)/F_STAR < 1e-8, \
    "ISTA must reach the reference optimum (check the 1/L step and lam/L threshold)"
assert np.array_equal(np.flatnonzero(np.abs(s_ista) > 1e-8), support), \
    "the exact support must be recovered at this sampling rate"
print(f"n={N}, k={K}, m={M}; L={L:.3f}, lambda={LAM:.4f}")
print(f"ISTA F = {F_ista:.10f}   reference F* = {F_STAR:.10f}   "
      f"gap = {(F_ista - F_STAR)/F_STAR:.2e}")

# %% [markdown]
# ## Task 2 — FISTA, and how much the momentum is worth (3 points)
#
# FISTA (Beck and Teboulle, 2009) evaluates the gradient at an *extrapolated*
# point:
#
# $$\boldsymbol{s}^{k+1}=\operatorname{SoftThresh}\!\Big(\boldsymbol{z}^k-\tfrac1L\nabla f(\boldsymbol{z}^k),\tfrac{\lambda}{L}\Big),\quad
#   t_{k+1}=\tfrac{1+\sqrt{1+4t_k^2}}{2},\quad
#   \boldsymbol{z}^{k+1}=\boldsymbol{s}^{k+1}+\tfrac{t_k-1}{t_{k+1}}(\boldsymbol{s}^{k+1}-\boldsymbol{s}^k)$$
#
# with $\boldsymbol{s}^0=\boldsymbol{z}^0=\mathbf{0}$, $t_0=1$.
#
# 1. Implement `fista(Phi, y, lam, n_iter, L)` with the same signature and
#    return convention as `ista`.
# 2. Write `iters_to(hist, tol)` returning the **1-based** iteration index at
#    which the relative objective gap `(hist - F_STAR)/F_STAR` first drops to
#    `tol` or below (`np.inf` if never).
# 3. Set `it_ista`, `it_fista` at `tol = 1e-6` and `speedup = it_ista/it_fista`.

# %%
### BEGIN SOLUTION
def fista(Phi, y, lam, n_iter, L):
    n = Phi.shape[1]
    s = np.zeros(n); z = s.copy(); t = 1.0
    hist = np.empty(n_iter)
    for k in range(n_iter):
        s_new = soft_threshold(z - (Phi.T @ (Phi @ z - y))/L, lam/L)
        t_new = 0.5*(1.0 + np.sqrt(1.0 + 4.0*t*t))
        z = s_new + ((t - 1.0)/t_new)*(s_new - s)
        s, t = s_new, t_new
        hist[k] = lasso_objective(Phi, y, s, lam)
    return s, hist


def iters_to(hist, tol):
    gap = (hist - F_STAR)/F_STAR
    hit = np.flatnonzero(gap <= tol)
    return int(hit[0]) + 1 if hit.size else np.inf


s_fista, hist_fista = fista(Phi, y, LAM, 4000, L)
it_ista = iters_to(hist_ista, 1e-6)
it_fista = iters_to(hist_fista, 1e-6)
speedup = it_ista/it_fista
### END SOLUTION

# %%
assert np.isfinite(it_ista) and np.isfinite(it_fista)
# same fixed point: both solve the SAME convex problem
assert np.linalg.norm(s_fista - s_ista)/np.linalg.norm(s_ista) < 1e-6, \
    "ISTA and FISTA must converge to the same LASSO solution"
# momentum must buy a material factor at identical cost per iteration
assert speedup >= 2.5, f"FISTA should need far fewer iterations, got x{speedup:.1f}"
# and both must respect their worst-case envelopes
kk = np.arange(1, 4001)
R0 = np.linalg.norm(s_fista)
gap_i = np.maximum(hist_ista - F_STAR, 1e-16)/F_STAR
gap_f = np.maximum(hist_fista - F_STAR, 1e-16)/F_STAR
assert np.max(gap_i/(L*R0**2/(2*kk)/F_STAR)) <= 1.0, "ISTA must respect O(1/k)"
assert np.max(gap_f/(2*L*R0**2/(kk + 1.0)**2/F_STAR)) <= 1.0, "FISTA must respect O(1/k^2)"
print(f"iterations to a 1e-6 relative gap: ISTA {it_ista}, FISTA {it_fista} "
      f"-> x{speedup:.1f} fewer, at identical cost per iteration")

# %% [markdown]
# ## Task 3 — recovery below the Nyquist rate (2 points)
#
# The provided signal is two tones at 97 and 337 Hz on a 1024-point grid, so
# $f_\max=337$ Hz and Nyquist demands **674 samples/s**. We take only
# `M_CS = 128` random samples. With `PSI` the DCT synthesis basis
# ($\boldsymbol{x}=\boldsymbol{\Psi}\boldsymbol{s}$), the sensing matrix is the
# selected rows, $\boldsymbol{\Phi}=\boldsymbol{\Psi}[\text{perm},:]$.
#
# 1. Solve the LASSO with your `fista` (3000 iterations,
#    `lam_cs = 0.005*max|Phi_cs.T @ y_cs|`, `L` from `np.linalg.norm(.,2)**2`).
# 2. Read off `supp_cs`, the indices with $|s_i|>10^{-2}\max|s|$.
# 3. **Debias**: re-fit by least squares on `supp_cs` only, into `s_cs_deb`.
# 4. Synthesise `x_rec = PSI @ s_cs_deb` and set `err_cs`, the relative error.

# %%
# provided data (do not modify)
N1D, T, F1, F2 = 1024, 1.0, 97.0, 337.0
t_grid = (np.arange(N1D) + 0.5)*T/N1D
x_full = np.cos(2*np.pi*F1*t_grid) + np.cos(2*np.pi*F2*t_grid)
PSI = idct(np.eye(N1D), norm='ortho', axis=0)          # x = PSI @ s
M_CS = 128
perm = np.sort(np.random.default_rng(3).choice(N1D, M_CS, replace=False))
y_cs = x_full[perm]
Phi_cs = PSI[perm, :]

### BEGIN SOLUTION
L_cs = np.linalg.norm(Phi_cs, 2)**2
lam_cs = 0.005*np.max(np.abs(Phi_cs.T @ y_cs))
s_cs, _ = fista(Phi_cs, y_cs, lam_cs, 3000, L_cs)
supp_cs = np.flatnonzero(np.abs(s_cs) > 1e-2*np.max(np.abs(s_cs)))
s_cs_deb = np.zeros(N1D)
s_cs_deb[supp_cs] = np.linalg.lstsq(Phi_cs[:, supp_cs], y_cs, rcond=None)[0]
x_rec = PSI @ s_cs_deb
err_cs = np.linalg.norm(x_rec - x_full)/np.linalg.norm(x_full)
### END SOLUTION

# %%
# the true DCT indices of the two tones are k = 2*f*T
true_supp = [int(2*F1*T), int(2*F2*T)]
assert sorted(supp_cs.tolist()) == true_supp, \
    f"support must be exactly {true_supp}, got {sorted(supp_cs.tolist())}"
assert err_cs < 1e-2, f"sub-Nyquist recovery must be essentially exact, got {err_cs:.2e}"
# the honest control: naive interpolation of the SAME samples fails badly
x_naive = np.interp(t_grid, t_grid[perm], y_cs)
err_naive = np.linalg.norm(x_naive - x_full)/np.linalg.norm(x_full)
assert err_naive > 50*err_cs, "compressed sensing must crush linear interpolation"
print(f"Nyquist {2*F2:.0f} samples/s, used {M_CS} -> {2*F2/M_CS:.1f}x below; "
      f"support {sorted(supp_cs.tolist())}")
print(f"relative error: CS {err_cs:.3e}   linear interpolation {err_naive:.3f}")

# %% [markdown]
# ## Task 4 — QR-pivot sensor placement (2 points)
#
# `PSI_R` below is a POD basis with `R` orthonormal columns, learned from
# training snapshots of a meandering wake field; `Xt` holds unseen snapshots
# (mean removed). With $m$ point sensors, gappy POD estimates the modal
# coefficients by least squares on $\boldsymbol{\Theta}=\mathbf{C}\boldsymbol{\Psi}_r$
# and synthesises the whole field.
#
# 1. `qr_sensors(m)` — the first `m` pivots of
#    `scipy.linalg.qr(PSI_R.T, pivoting=True)` for `m == R`, and of
#    `scipy.linalg.qr(PSI_R @ PSI_R.T, pivoting=True)` for `m > R`.
#    Return them sorted.
# 2. `gappy_error(sensors)` — returns `(relative reconstruction error over all
#    of Xt with NOISE added at the sensors, cond(Theta))`. Use
#    `np.linalg.lstsq` and the provided `noise_rng` exactly as written below.
# 3. Set `err_qr, cond_qr` at `m = R`, and `err_rand, cond_rand` as arrays over
#    the `NDRAW` random sensor sets produced by `random_sensors(m, d)`.

# %%
# provided field, POD basis and noise model (do not modify)
NX, NY, NT, NTRAIN = 48, 30, 320, 220
xg = np.linspace(0.0, 12.0, NX); yg = np.linspace(-3.0, 3.0, NY)
XX, YY = np.meshgrid(xg, yg)
N_SP = NX*NY
tt = np.linspace(0, 160.0, NT)
snaps = np.empty((NT, N_SP))
for it, tv in enumerate(tt):
    u = np.full_like(XX, 10.0)
    for (xp, yp, amp, ma, mf, bf) in [(1.0, -1.3, 0.42, 0.30, 0.070, 0.031),
                                      (1.0, 1.4, 0.38, 0.26, 0.052, 0.043),
                                      (6.0, 0.0, 0.34, 0.34, 0.038, 0.026)]:
        down = XX - xp
        w = 0.45 + 0.075*np.maximum(down, 0.0)
        yc = yp + ma*np.sin(2*np.pi*mf*tv + xp)
        a = amp*(1.0 + 0.18*np.sin(2*np.pi*bf*tv))
        u -= 10.0*a*(down > 0)*(1.0 - np.exp(-np.maximum(down, 0.0)/2.5)) \
             * np.exp(-(YY - yc)**2/(2*w**2))*(0.45/w)
    u += 0.25*np.sin(2*np.pi*(0.09*tv - XX/4.0))*np.exp(-(YY/2.2)**2)
    snaps[it] = u.ravel()

train, test = snaps[:NTRAIN], snaps[NTRAIN:]
u_mean = train.mean(0)
_, sv, VT = np.linalg.svd(train - u_mean, full_matrices=False)
R = int(np.searchsorted(np.cumsum(sv**2)/np.sum(sv**2), 0.99999) + 1)
PSI_R = VT[:R].T                                   # (n, r), orthonormal columns
Xt = test - u_mean
NOISE = 0.02*train.std()
NDRAW = 60


def random_sensors(m, d):
    return np.sort(np.random.default_rng(900 + 31*m + d).choice(N_SP, m, replace=False))


### BEGIN SOLUTION
piv_exact = qr(PSI_R.T, pivoting=True)[2]
piv_over = qr(PSI_R @ PSI_R.T, pivoting=True)[2]


def qr_sensors(m):
    return np.sort(piv_exact[:R] if m == R else piv_over[:m])


def gappy_error(sensors):
    Theta = PSI_R[sensors, :]
    noise_rng = np.random.default_rng(11)
    Y = Xt[:, sensors].T + noise_rng.normal(0, NOISE, (len(sensors), Xt.shape[0]))
    A = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    Xh = (PSI_R @ A).T
    return (np.linalg.norm(Xh - Xt)/np.linalg.norm(Xt), np.linalg.cond(Theta))


err_qr, cond_qr = gappy_error(qr_sensors(R))
_res = [gappy_error(random_sensors(R, d)) for d in range(NDRAW)]
err_rand = np.array([e for e, _ in _res])
cond_rand = np.array([c for _, c in _res])
### END SOLUTION

# %%
proj_err = np.linalg.norm(Xt - (Xt @ PSI_R) @ PSI_R.T)/np.linalg.norm(Xt)
assert len(qr_sensors(R)) == R and len(set(qr_sensors(R).tolist())) == R, \
    "qr_sensors must return R distinct locations"
assert len(qr_sensors(R + 12)) == R + 12, "the oversampled branch must return m sensors"
# placement beats a typical random draw on BOTH the error and the mechanism
assert np.median(err_rand)/err_qr > 3.0, \
    f"QR must beat the random median at m=r, got x{np.median(err_rand)/err_qr:.1f}"
assert np.median(cond_rand)/cond_qr > 5.0, \
    f"QR must be far better conditioned, got x{np.median(cond_rand)/cond_qr:.1f}"
assert np.mean(err_rand < err_qr) < 0.05, "almost no random draw should beat QR"
# and with the noise off, r well-placed points reach the projection floor
err_clean = np.linalg.norm(
    Xt - (PSI_R @ np.linalg.lstsq(PSI_R[qr_sensors(R), :], Xt[:, qr_sensors(R)].T,
                                  rcond=None)[0]).T)/np.linalg.norm(Xt)
assert err_clean < 5*proj_err, "noise-free QR must essentially reach the projection floor"
print(f"r = {R} modes; projection floor {proj_err:.2e}")
print(f"m = r = {R}: QR error {err_qr:.4f} (cond {cond_qr:.1f}) vs random median "
      f"{np.median(err_rand):.4f} (cond {np.median(cond_rand):.1f}) "
      f"-> x{np.median(err_rand)/err_qr:.1f} and x{np.median(cond_rand)/cond_qr:.1f}")
print(f"random draws beating QR: {np.mean(err_rand < err_qr):.3f}; "
      f"noise-free QR {err_clean:.2e}")

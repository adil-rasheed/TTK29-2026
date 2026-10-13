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
# # Assignment — heat PINN
#
# Solution cells use `### BEGIN/END SOLUTION` (nbgrader). Builds on `notebooks/pinn_autodiff.ipynb`
# (Task 0) and `notebooks/pinn_heat_pde.ipynb` (Tasks 1–2).

# %%
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..','..')))
import numpy as np, torch, matplotlib.pyplot as plt
from pyutils import pinn_core as pc
from pyutils.pinn_core import MLP, grad, train
torch.set_default_dtype(torch.float64); torch.manual_seed(0); pc.set_plot_style()
rng = np.random.default_rng(0)   # all NumPy randomness goes through this generator


# %% [markdown]
# ## Task 0 — derivatives by hand and by automatic differentiation
#
# Builds on `notebooks/pinn_autodiff.ipynb`. Use raw `torch.autograd.grad`
# calls here, not the `grad` helper.
#
# **(a)** For the one-neuron network $u=v\tanh(wx+b)+c$ with $v=-0.8$,
# $w=3$, $b=0.2$, $c=0.1$, evaluate at $x=0.25$:
# `du_hand`, `d2u_hand` from the chain-rule formulas
# $u_x=v\,w\,(1-s^2)$ and $u_{xx}=-2\,v\,w^2\,s\,(1-s^2)$ with $s=\tanh(wx+b)$,
# and `du_ad`, `d2u_ad` with `torch.autograd.grad`.
#
# **(b)** Write `heat_residual(ufun, x, t, alpha)` returning $u_t-\alpha u_{xx}$
# for any differentiable `ufun(x, t)`. The checks apply it to the exact
# solution $e^{-\alpha\pi^2t}\sin\pi x$ (residual must vanish) and compare it with
# the `grad` helper on a random network.

# %%
v_, w_, b_, c_, x0 = -0.8, 3.0, 0.2, 0.1, 0.25
du_hand = d2u_hand = du_ad = d2u_ad = None
### BEGIN SOLUTION
s_ = np.tanh(w_ * x0 + b_)
du_hand = v_ * w_ * (1 - s_**2)
d2u_hand = -2 * v_ * w_**2 * s_ * (1 - s_**2)
xt = torch.tensor([x0], requires_grad=True)
ut = v_ * torch.tanh(w_ * xt + b_) + c_
(du,) = torch.autograd.grad(ut, xt, create_graph=True)
(d2u,) = torch.autograd.grad(du, xt)
du_ad, d2u_ad = du.item(), d2u.item()
### END SOLUTION
print(f'du/dx   hand {du_hand:.10f}   AD {du_ad:.10f}')
print(f'd2u/dx2 hand {d2u_hand:.10f}   AD {d2u_ad:.10f}')
assert abs(du_hand - du_ad) < 1e-12 and abs(d2u_hand - d2u_ad) < 1e-12

# %%
def heat_residual(ufun, x, t, alpha):
    ### BEGIN SOLUTION
    u = ufun(x, t)
    (u_t,) = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)
    (u_x,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
    (u_xx,) = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)
    return u_t - alpha * u_xx
    ### END SOLUTION

xc = torch.rand(100, 1, requires_grad=True); tc = torch.rand(100, 1, requires_grad=True)
r_exact = heat_residual(lambda x, t: torch.exp(-0.1 * np.pi**2 * t) * torch.sin(np.pi * x), xc, tc, 0.1)
assert r_exact.abs().max().item() < 1e-12, "the exact solution must give a zero residual"
net0 = MLP([2, 16, 16, 1])
r_net = heat_residual(lambda x, t: net0(torch.cat([x, t], 1)), xc, tc, 0.1)
u0_ = net0(torch.cat([xc, tc], 1))
r_ref = grad(u0_, tc) - 0.1 * grad(grad(u0_, xc), xc)
assert torch.allclose(r_net, r_ref), "must agree with the grad helper"
print('Task 0 passed')

# %% [markdown]
# ## Task 1 — a faster-decaying initial condition
#
# Solve the forward heat PINN with $u(x,0)=\sin(2\pi x)$ (zero Dirichlet ends), $\alpha=0.1$, $T=0.3$. The exact decay is $e^{-\alpha(2\pi)^2 t}$. Train the PINN and report the max-abs error against the exact solution.

# %%
alpha, T = 0.1, 0.3
exact = lambda x,t: np.exp(-alpha*(2*np.pi)**2*t)*np.sin(2*np.pi*x)
max_err = None
### BEGIN SOLUTION
net = MLP([2,32,32,32,1])
def loss_fn():
    x=torch.rand(2000,1).requires_grad_(True); t=(torch.rand(2000,1)*T).requires_grad_(True)
    u=net(torch.cat([x,t],1)); u_t=grad(u,t); u_xx=grad(grad(u,x),x)
    r=u_t-alpha*u_xx
    xi=torch.rand(200,1); Li=((net(torch.cat([xi,torch.zeros_like(xi)],1))-torch.sin(2*np.pi*xi))**2).mean()
    tb=torch.rand(200,1)*T
    Lb=(net(torch.cat([torch.zeros_like(tb),tb],1))**2).mean()+(net(torch.cat([torch.ones_like(tb),tb],1))**2).mean()
    return (r**2).mean()+10*Li+10*Lb
train(loss_fn, net.parameters(), steps=4000, lr=2e-3)
xg,tg=np.meshgrid(np.linspace(0,1,60),np.linspace(0,T,60))
XX=torch.tensor(np.c_[xg.ravel(),tg.ravel()])
with torch.no_grad(): uu=net(XX).numpy().reshape(xg.shape)
max_err=float(np.abs(uu-exact(xg,tg)).max())
### END SOLUTION
print('max abs error:', max_err)
assert max_err is not None and max_err < 0.05

# %% [markdown]
# ## Task 2 — inverse for a different diffusivity
#
# With true $\alpha=0.05$ and noisy data from the exact $\sin(\pi x)$ solution, recover $\alpha$ to within 15%.

# %%
alpha_true, T = 0.05, 1.0
ex = lambda x,t: np.exp(-alpha_true*np.pi**2*t)*np.sin(np.pi*x)
alpha_hat = None
### BEGIN SOLUTION
net=MLP([2,32,32,32,1]); la=torch.tensor(np.log(0.2),requires_grad=True)
nd=60; xd=rng.random(nd); td=rng.random(nd)*T
Xd=torch.tensor(np.c_[xd,td]); Ud=torch.tensor(ex(xd,td)+0.01*rng.standard_normal(nd)).reshape(-1,1)
def loss():
    a=torch.exp(la); x=torch.rand(2000,1).requires_grad_(True); t=(torch.rand(2000,1)*T).requires_grad_(True)
    u=net(torch.cat([x,t],1)); r=grad(u,t)-a*grad(grad(u,x),x)
    return (r**2).mean()+100*((net(Xd)-Ud)**2).mean()
train(loss, list(net.parameters())+[la], steps=4000, lr=2e-3)
alpha_hat=float(torch.exp(la).detach())
### END SOLUTION
print(f'recovered alpha={alpha_hat:.4f} (true {alpha_true})')
assert abs(alpha_hat-alpha_true)/alpha_true < 0.15

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
# # Assignment 0: environment check
#
# Worth 0 course points and 100% of your future sanity. If every cell runs
# and every assert passes, your machine is ready for the whole course.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

# %% [markdown]
# ## Task 1 — the scientific stack

# %%
import numpy as np
import scipy
import matplotlib
import torch

print("python     ", sys.version.split()[0])
print("numpy      ", np.__version__)
print("scipy      ", scipy.__version__)
print("matplotlib ", matplotlib.__version__)
print("torch      ", torch.__version__)

assert sys.version_info >= (3, 11), "please use the course conda environment (python>=3.11)"
assert int(np.__version__.split('.')[0]) >= 2

# %% [markdown]
# ## Task 2 — the course utilities

# %%
from pyutils import set_plot_style, save_dual, fg
from pyutils.pinn_core import MLP, grad, train
set_plot_style()
print("pyutils imports OK")

# %% [markdown]
# ## Task 3 — a 30-second sanity computation
#
# Solve $\dot u = -u$ with RK4 and check against $e^{-t}$; then take one
# torch gradient. If both work, every notebook in the course will.

# %%
dt, n = 0.01, 100
u = 1.0
for _ in range(n):
    k1 = -u; k2 = -(u+dt/2*k1); k3 = -(u+dt/2*k2); k4 = -(u+dt*k3)
    u = u + dt/6*(k1+2*k2+2*k3+k4)
assert abs(u - np.exp(-1.0)) < 1e-8, "RK4 sanity check failed"

x = torch.tensor([2.0], requires_grad=True)
y = x**3
(dy,) = torch.autograd.grad(y, x)
assert abs(float(dy) - 12.0) < 1e-12, "autograd sanity check failed"

print("all good -- see you in Lecture 1")

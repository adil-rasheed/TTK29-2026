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
# # Assignment — wind/wake PINN
#
# Builds on `notebooks/pinn_wind_pde.ipynb`.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..','..')))
import numpy as np, torch, matplotlib.pyplot as plt
from pyutils import pinn_core as pc
from pyutils.pinn_core import MLP, grad, train
torch.set_default_dtype(torch.float64); torch.manual_seed(0); pc.set_plot_style()
rng = np.random.default_rng(0)   # all NumPy randomness goes through this generator


# %% [markdown]
# ## Task — recover a different eddy viscosity & check held-out error
#
# With true $\nu_t=0.8$, recover it from sparse data (within 15%) and report the reconstruction RMSE on a held-out test grid.
#
# Draw the measurements with the seeded generator `rng`. The recovered value
# depends on the noise draw and the training seed: in `pinn_wind_pde` the error
# in $\nu_t$ ranged from 4% to 19% over five seeds, always on the low side.
# If your estimate misses the 15% target, check the final loss first.

# %%
U,nu_true=8.0,0.8; D=nu_true/U
wake=lambda x,y: 1/np.sqrt(4*np.pi*D*x)*np.exp(-y**2/(4*D*x))
x0,x1,W=1.0,6.0,3.0
nu_hat=rmse=None
### BEGIN SOLUTION
nd=120; xd=rng.uniform(x0,x1,nd); yd=rng.uniform(-W,W,nd)
Xd=torch.tensor(np.c_[xd,yd]); Dd=torch.tensor(wake(xd,yd)+0.003*rng.standard_normal(nd)).reshape(-1,1)
sx=lambda x:(x-x0)/(x1-x0); sy=lambda y:y/W
net=MLP([2,40,40,40,1]); lnu=torch.tensor(np.log(2.0),requires_grad=True)
df=lambda x,y: net(torch.cat([sx(x),sy(y)],1))
def loss():
    nu=torch.exp(lnu); x=(torch.rand(3000,1)*(x1-x0)+x0).requires_grad_(True); y=(torch.rand(3000,1)*2*W-W).requires_grad_(True)
    d=df(x,y); r=U*grad(d,x)-nu*grad(grad(d,y),y)
    return (r**2).mean()+100*((df(Xd[:,:1],Xd[:,1:])-Dd)**2).mean()
train(loss, list(net.parameters())+[lnu], steps=5000, lr=2e-3)
nu_hat=float(torch.exp(lnu).detach())
xg,yg=np.meshgrid(np.linspace(x0,x1,50),np.linspace(-W,W,50)); XX=torch.tensor(np.c_[xg.ravel(),yg.ravel()])
with torch.no_grad(): dp=df(XX[:,:1],XX[:,1:]).numpy().reshape(xg.shape)
rmse=float(np.sqrt(np.mean((dp-wake(xg,yg))**2)))
### END SOLUTION
print(f'nu_t={nu_hat:.4f} (true {nu_true}),  held-out RMSE={rmse:.4e}')
assert abs(nu_hat-nu_true)/nu_true < 0.15

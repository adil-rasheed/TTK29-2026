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
# # Assignment — turn a learned correction into an equation
#
# A physics model of the vessel omits the quadratic drag. A hybrid model fixes this by adding a neural network
# $\boldsymbol{f}_\theta(\boldsymbol{\nu})$ to the equations and training it on data until the simulation matches the
# measurements (a *corrective source term*). The network works, but it cannot say what it learned. Here you are given
# samples of that learned correction over the visited states; **discover the symbolic drag law** by sparse regression ---
# closing the data → black box → correction → **equation** loop.

# %%
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..','..')))
from pyutils import set_plot_style; set_plot_style(); np.random.seed(0)
# samples of the (true) unmodeled drag the correction network learned, over states nu=(u,v,r)
N=2000; nu=np.random.uniform(-2,2,size=(N,3))
u,v,r=nu[:,0],nu[:,1],nu[:,2]
f_corr=np.stack([-3.0*np.abs(u)*u, -8.0*np.abs(v)*v, -2.0*np.abs(r)*r],1)  # quadratic drag
f_corr+=0.02*np.random.randn(*f_corr.shape)  # small model/sampling noise
def stlsq(Theta,Y,lam,n_iter=20):
    norm=np.linalg.norm(Theta,axis=0); Tn=Theta/norm
    Xi=np.linalg.lstsq(Tn,Y,rcond=None)[0]
    Y2=Y if Y.ndim>1 else Y.reshape(-1,1); Xi=Xi.reshape(Tn.shape[1],-1)
    for _ in range(n_iter):
        small=np.abs(Xi)<lam; Xi[small]=0
        for j in range(Xi.shape[1]):
            big=~small[:,j]
            if big.any(): Xi[big,j]=np.linalg.lstsq(Tn[:,big],Y2[:,j],rcond=None)[0]
    return (Xi/norm[:,None])

# %% [markdown]
# ## Task — recover the drag law per component
#
# Build a library $\{u,v,r,|u|u,|v|v,|r|r,u^3,v^3,r^3\}$ and sparse-regress each component of $\boldsymbol{f}_\theta$. Recover the three quadratic-drag coefficients $(-3,-8,-2)$ to within 0.3.

# %%
c_u=c_v=c_r=None
### BEGIN SOLUTION
lib={'u':u,'v':v,'r':r,'|u|u':np.abs(u)*u,'|v|v':np.abs(v)*v,'|r|r':np.abs(r)*r,
     'u^3':u**3,'v^3':v**3,'r^3':r**3}
names=list(lib); Theta=np.stack([lib[n] for n in names],1)
Xi=stlsq(Theta, f_corr, lam=0.2)
for j,comp in enumerate(['f_u','f_v','f_r']):
    print(f'{comp} =', ' '.join(f'{Xi[i,j]:+.2f}*{names[i]}' for i in range(len(names)) if abs(Xi[i,j])>1e-6))
c_u=Xi[names.index('|u|u'),0]; c_v=Xi[names.index('|v|v'),1]; c_r=Xi[names.index('|r|r'),2]
### END SOLUTION
print(f'recovered drag coeffs: {c_u:.2f}, {c_v:.2f}, {c_r:.2f}  (true -3, -8, -2)')
assert abs(c_u+3)<0.3 and abs(c_v+8)<0.3 and abs(c_r+2)<0.3

# %% [markdown]
# **You just closed the hybrid-modeling loop:** the black-box correction is now an explicit, reusable physical term $\boldsymbol{f}=[-3|u|u,\,-8|v|v,\,-2|r|r]$ that could be added back to the physics model.

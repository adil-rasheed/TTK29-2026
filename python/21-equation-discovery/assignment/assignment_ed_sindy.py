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
# # Assignment — discover the Duffing oscillator with SINDy
#
# Solution cells use `### BEGIN/END SOLUTION` (nbgrader). The damped Duffing oscillator is $\dot x=v,\ \dot v=-\delta v-\alpha x-\beta x^3$. From data, recover the $\dot v$ equation and its coefficients.

# %%
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..','..')))
from pyutils import set_plot_style; set_plot_style(); np.random.seed(0)
delta,alpha,beta=0.3,-1.0,1.0   # ground truth
def deriv(s): x,v=s; return np.array([v, -delta*v-alpha*x-beta*x**3])
def rk4(s0,dt,n):
    S=[s0.copy()]; s=s0.copy()
    for _ in range(n):
        k1=deriv(s);k2=deriv(s+0.5*dt*k1);k3=deriv(s+0.5*dt*k2);k4=deriv(s+dt*k3)
        s=s+dt/6*(k1+2*k2+2*k3+k4); S.append(s.copy())
    return np.array(S)
dt=0.01; S=np.vstack([rk4(np.array([a,0.0]),dt,3000) for a in (1.5,-1.0,0.5)])
x,v=S[:,0],S[:,1]
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
# ## Task — recover $\dot v=-\delta v-\alpha x-\beta x^3$
#
# Build a library $\{x,v,x^2,x^3,xv,v^2\}$, compute $\dot v$ analytically, run STLSQ, and recover the three coefficients to within 0.05.

# %%
vdot=-delta*v-alpha*x-beta*x**3
co_x=co_v=co_x3=None
### BEGIN SOLUTION
lib={'x':x,'v':v,'x^2':x**2,'x^3':x**3,'xv':x*v,'v^2':v**2}
names=list(lib); Theta=np.stack([lib[n] for n in names],1)
xi=stlsq(Theta, vdot, lam=0.05).ravel()
co_x=xi[names.index('x')]; co_v=xi[names.index('v')]; co_x3=xi[names.index('x^3')]
print('vdot =', ' '.join(f'{xi[i]:+.3f}*{names[i]}' for i in range(len(names)) if abs(xi[i])>1e-6))
### END SOLUTION
print(f'coeffs  x:{co_x:+.3f}(true {-alpha})  v:{co_v:+.3f}(true {-delta})  x^3:{co_x3:+.3f}(true {-beta})')
assert abs(co_x-(-alpha))<0.05 and abs(co_v-(-delta))<0.05 and abs(co_x3-(-beta))<0.05

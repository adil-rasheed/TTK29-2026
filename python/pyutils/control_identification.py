"""Measured dictionary selection and scalar grey-box inverse PINNs."""
import time
import numpy as np
from .control_models import library,stlsq,KoopmanLift,LIFT_NAMES
from .benchmarks.vessel import VesselParams,step


def select_dictionary(Z,U,train,validation):
    """SINDy selects from the 27-observable ladder, validation selects threshold.

    State observables are mandatory for the readout. Only training transitions
    enter either regression. Validation measures physical one-step increments.
    """
    X=np.c_[library(Z[train],True),U[train]];Y=Z[train+1]-Z[train]
    trials=[];models=[]
    for threshold in (.0001,.001,.003,.01,.03):
        W=stlsq(X,Y,threshold)
        chosen=sorted(set(range(6))|set(np.flatnonzero(np.any(W[:27]!=0,axis=1))))
        model=KoopmanLift(chosen).fit(Z,U,train)
        err=float(np.linalg.norm(model.predict(Z[validation],U[validation])-Z[validation+1])/np.linalg.norm(Z[validation+1]-Z[validation]))
        trials.append({'threshold':threshold,'indices':chosen,'names':[LIFT_NAMES[j] for j in chosen],'validation_closure':err})
        models.append(model)
    best=int(np.argmin([t['validation_closure'] for t in trials]))
    return models[best],{'selected':trials[best],'candidates':trials}


def scalar_inverse_pinn(Z,U,train,dt=.5,epochs=1200,seed=0):
    """Three scalar inverse PINNs, with measured coupling treated as forcing.

    A neural correction s(1-s)g(s,nu_k,nu_{k+1}) interpolates each measured
    interval and vanishes at both endpoints. Autodiff gives d nu / dt at the
    midpoint. Learn positive linear/quadratic damping from the scalar momentum
    residual. This is parameter identification, not a 3-DOF forward PINN.
    The correction is regularised because noisy endpoint interpolation alone
    does not identify both a free trajectory and coefficients uniquely.
    """
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(seed);start=time.perf_counter()
    p=VesselParams();mass=np.diag(p.M());mid=(Z[train,3:]+Z[train+1,3:])/2
    u,v,r=mid.T;coupling=np.c_[-mass[1]*v*r,mass[0]*u*r,(mass[1]-mass[0])*u*v]
    output=[];fits=[]
    for j in range(3):
        y0=torch.tensor(Z[train,3+j:4+j],dtype=torch.float64)
        y1=torch.tensor(Z[train+1,3+j:4+j],dtype=torch.float64)
        force=torch.tensor((U[train,j]-coupling[:,j])[:,None],dtype=torch.float64)
        net=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1)).double()
        initial=np.array([[p.Du,p.Xuu],[p.Dv,p.Yvv],[p.Dr,p.Nrr]])[j]
        logc=nn.Parameter(torch.tensor(np.log(initial),dtype=torch.float64))
        opt=torch.optim.Adam(list(net.parameters())+[logc],lr=.015)
        for _ in range(epochs):
            s=torch.full_like(y0,.5,requires_grad=True)
            correction=.001*s*(1-s)*net(torch.cat([s,y0,y1],1))
            y=y0+s*(y1-y0)+correction
            dy=torch.autograd.grad(y.sum(),s,create_graph=True)[0]/dt
            coeff=logc.exp();res=(mass[j]*dy+coeff[0]*y+coeff[1]*y.abs()*y-force)
            loss=(res**2).mean()+1e6*(correction**2).mean()
            opt.zero_grad();loss.backward();opt.step()
        fits.append(logc.exp().detach().numpy());output.append(float(loss.detach()))
    co=np.array(fits)
    fitted=VesselParams(Du=co[0,0],Dv=co[1,0],Dr=co[2,0],Xuu=co[0,1],Yvv=co[1,1],Nrr=co[2,1])
    return fitted,{'coefficients':dict(zip(['Du','Dv','Dr','Xuu','Yvv','Nrr'],np.r_[co[:,0],co[:,1]].tolist())),
                   'physics_loss':output,'training_seconds':time.perf_counter()-start,
                   'method':'three scalar neural temporal interpolants, endpoint anchored, momentum residual'}

"""Empirical domains, conditional tubes and the learned-controller shield."""
import time
import numpy as np
from scipy.linalg import solve_discrete_are
from .control_models import LinearAtRest
from .control_experiments import controller
from .control import MPCSolveError
from .benchmarks.vessel import step


def domain_evaluation(models,Z,U,idx):
    rows={}
    for name,model in models.items():
        err=np.linalg.norm(model.predict(Z[idx],U[idx])-Z[idx+1],axis=1)
        h=model.leverage(Z[idx],U[idx])/model.hstar
        rows[name]={'rmse':float(np.sqrt(np.mean(err**2))),'max_error':float(err.max()),
                    'leverage':h.tolist(),'errors':err.tolist(),
                    'outside_fraction':float(np.mean(h>1))}
    return rows


def tube_radius(model,Z,U,idx,horizon=12):
    """Empirical componentwise max on independent disturbed calibration pairs.

    Local linear A-BK propagation, conditional on this bound and linearisation
    remaining valid. This is NOT a nonlinear robust-invariance certificate.
    """
    truth=step(Z[idx],U[idx],current=(0.,12.))
    w=np.max(abs(truth-model.predict(Z[idx],U[idx])),axis=0)
    z=np.array([0.,1.45,0.,.45,0.,0.]);A,B=model.jacobians(z,np.zeros(3))
    Q=np.diag([2.,60.,10.,1.,1.,1.]);R=np.eye(3)*.002
    try:
        P=solve_discrete_are(A,B,Q,R);K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
        AK=A-B@K;rho=float(max(abs(np.linalg.eigvals(AK))))
        row=np.eye(6)[1];delta=[];acc=0.
        for _ in range(horizon):acc+=abs(row)@w;delta.append(float(acc));row=row@AK
        return {'w_bar':w.tolist(),'delta':delta,'local_closed_loop_radius':rho,'valid':rho<1}
    except np.linalg.LinAlgError:
        return {'w_bar':w.tolist(),'delta':None,'valid':False,'reason':'local Riccati solve failed'}


def tube_mission(model,delta=None,nsteps=200):
    N=12;dt=.5;Q=np.diag([2.,60.,10.,1.,1.,1.]);R=np.eye(3)*.002;umax=np.array([40.,35.,20.])
    C=np.eye(6)[1:2];d=np.zeros(N) if delta is None else np.asarray(delta)
    if np.max(d)>=1.5:return {'feasible_design':False,'reason':'tightening consumes the channel','cost':None}
    ctl=controller(model,Q,R,N,umax,(C,-np.inf,(1.5-d)[:,None]))
    refs=np.c_[.45*np.arange(nsteps+N+1)*dt,np.full(nsteps+N+1,1.45),np.zeros(nsteps+N+1),np.full(nsteps+N+1,.45),np.zeros((nsteps+N+1,2))]
    z=np.array([0.,1.,0.,.45,0.,0.]);states=[];inputs=[];ms=[];failed=0
    base=LinearAtRest();P=solve_discrete_are(base.A,base.B,Q,R);K=np.linalg.solve(R+base.B.T@P@base.B,base.B.T@P@base.A)
    for k in range(nsteps):
        start=time.perf_counter()
        try:u,_=ctl.solve(z,refs[k+1:k+N+1])
        except (MPCSolveError,ValueError,np.linalg.LinAlgError):
            failed+=1;e=z-np.array([z[0],0,0,.15,0,0]);u=np.clip(-K@e,-umax,umax)
        ms.append(1e3*(time.perf_counter()-start));states.append(z.copy());inputs.append(u)
        z=step(z,u,current=(0.,9+16*np.sin(2*np.pi*(k*dt+57)/30)))
    Z,V=np.array(states),np.array(inputs);e=Z-refs[:nsteps];vi=np.maximum(Z[:,1]-1.5,0)
    return {'feasible_design':True,'cost':float(np.einsum('ti,ij,tj->',e,Q,e)+np.einsum('ti,ij,tj->',V,R,V)),
            'state_violation_count':int((vi>1e-4).sum()),'state_violation_worst_m':float(vi.max()),
            'fallback_count':failed,'solve_median_ms':float(np.median(ms)),'solve_max_ms':float(max(ms))}


def path_state(x,V):
    kw=2*np.pi/18;c=1.5*kw*np.cos(kw*x);psi=np.arctan(c)
    dpsi=-1.5*kw**2*np.sin(kw*x)/(1+c*c)
    return np.array([x,1.5*np.sin(kw*x),psi,V*np.sqrt(1+c*c),0.,dpsi*V])


def shield_mission(model,shielded=True,v1=1.75,nsteps=320):
    """Original Module 30 slalom, learned MPC and unchanged low-speed LQR.

    Monitor features are model-specific and fitted on the shared train segment.
    Trip at 3 h*, release at h*, with 2/12 sample dwell; record breaches rather
    than asserting a global safety certificate for a finite-data monitor.
    """
    N=15;dt=.5;ts=np.arange(nsteps+N+2)*dt;V=.45+(v1-.45)*np.clip(ts/160,0,1)
    x=np.r_[0,np.cumsum(.5*(V[1:]+V[:-1])*dt)];refs=np.array([path_state(a,b) for a,b in zip(x,V)])
    Q=np.diag([5.,60.,20.,1.,1.,1.]);R=np.eye(3)*.002;umax=np.array([45.,10.,10.])
    ctl=controller(model,Q,R,N,umax);base=LinearAtRest();Qfb=np.diag([0.,30.,15.,4.,4.,2.]);Rfb=np.eye(3)*.05
    P=solve_discrete_are(base.A,base.B,Qfb,Rfb);K=np.linalg.solve(Rfb+base.B.T@P@base.B,base.B.T@P@base.A)
    z=refs[0].copy();up=np.zeros(3);on=False;hi=lo=switches=failures=0
    zs=[];us=[];hs=[];active=[];ms=[]
    for k in range(nsteps):
        start=time.perf_counter();h=float(model.leverage(z,up)/model.hstar)
        hi=hi+1 if h>3 else 0;lo=lo+1 if h<1 else 0;was=on
        if shielded:
            if not on and hi>=2:on=True
            elif on and lo>=12:on=False
        switches+=int(was!=on)
        def fallback():
            e=z-path_state(z[0],.45);e[0]=0
            return np.clip(-K@e,-umax,umax)
        if on:u=fallback()
        else:
            try:u,_=ctl.solve(z,refs[k+1:k+N+1])
            except (MPCSolveError,ValueError,np.linalg.LinAlgError):u=fallback();failures+=1
        ms.append(1e3*(time.perf_counter()-start));zs.append(z.copy());us.append(u);hs.append(h);active.append(on)
        up=u;z=step(z,u)
    Z,U=np.array(zs),np.array(us);e=Z-refs[:nsteps];vi=np.maximum(abs(Z[:,1])-1.85,0)
    cross=Z[:,1]-1.5*np.sin(2*np.pi*Z[:,0]/18)
    return {'cost':float(np.einsum('ti,ij,tj->',e,Q,e)+np.einsum('ti,ij,tj->',U,R,U)),
            'cross_track_rms_m':float(np.sqrt(np.mean(cross**2))),
            'violation_count':int((vi>1e-6).sum()),'violation_worst_m':float(vi.max()),
            'shield_fraction':float(np.mean(active)),'switches':switches,'distance_m':float(Z[-1,0]),
            'solver_failures':failures,'solve_median_ms':float(np.median(ms)),'solve_max_ms':float(max(ms)),
            'trace':{'z':Z.tolist(),'u':U.tolist(),'leverage_ratio':hs,'active':active}}

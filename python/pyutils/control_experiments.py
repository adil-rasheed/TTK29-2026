"""Shared task and scoring protocol for Modules 27--30 (no work at import)."""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import solve_discrete_are
from .benchmarks.vessel import step,commissioning_record,commissioning_split
from .control import CondensedMPC,SuccessiveLinearisationMPC,MPCSolveError
from .control_models import LinearAtRest,DMDcModel
DT,N_H,NSTEP=.5,20,360
Q=np.diag([20.,20.,10.,1.,1.,1.]);R=np.eye(3)*.002
UMAX=np.array([30.,22.,15.]);CURRENT=np.array([14.,-16.])
SHIFTED_CURRENT=1.5*np.array([-CURRENT[1],CURRENT[0]])
SEEDS=(0,1,2);RECORD_SEED=42

def reference(t):
    t=np.asarray(t);xd=.4;yd=8*(2*np.pi/120)*np.cos(2*np.pi*t/120)
    ydd=-8*(2*np.pi/120)**2*np.sin(2*np.pi*t/120);speed=np.hypot(xd,yd)
    return np.stack([xd*t,8*np.sin(2*np.pi*t/120),np.arctan2(yd,xd),speed,np.zeros_like(t),xd*ydd/speed**2],-1)

def protocol():
    return {'record_seed':RECORD_SEED,'model_seeds':list(SEEDS),'samples':900,
            'split':{k:[int(v[0]),int(v[-1])+1] for k,v in commissioning_split().items()},
            'split_units':'transition indices, end exclusive; 20-step guards',
            'dt_s':DT,'horizon':N_H,'steps':NSTEP,'Q_diagonal':Q.diagonal().tolist(),
            'R_diagonal':R.diagonal().tolist(),'input_bounds':UMAX.tolist(),
            'nominal_current_N':CURRENT.tolist(),'shifted_current_N':SHIFTED_CURRENT.tolist(),
            'offset_free':'same exponential filter gamma=.5 for every model; no current given to controller',
            'state_error':'RMS Euclidean norm of [m,m,rad,m/s,m/s,rad/s], unscaled',
            'rollout':'20 recursive one-step predictions within held-out segment',
            'cost':'sum e_k^T Q e_k + u_k^T R u_k, before transition, no terminal evaluation cost',
            'constraints':'actuator boxes; no state constraint in curved-transit comparison',
            'training_data':'all Part VI models are refits, not Part V checkpoints'}

def prediction_metrics(model,Z,U,idx):
    e=model.predict(Z[idx],U[idx])-Z[idx+1]
    starts=idx[:-(N_H-1)][::N_H];z=Z[starts].copy();errors=[]
    for j in range(N_H):
        z=model.predict(z,U[starts+j]);errors.append(z-Z[starts+j+1])
    return {'one_step_rmse':float(np.sqrt(np.mean(np.sum(e**2,axis=-1)))),
            'rollout_20_rmse':float(np.sqrt(np.mean(np.sum(np.array(errors)**2,axis=-1)))),
            'one_step_relative_increment':float(np.linalg.norm(e)/np.linalg.norm(Z[idx+1]-Z[idx])),
            'test_transitions':len(idx),'rollout_starts':starts.tolist()}

def controller(model,Q_=Q,R_=R,N=N_H,umax=UMAX,state_constraints=None):
    base=LinearAtRest();P=solve_discrete_are(base.A,base.B,Q_,R_)
    if type(model) in (LinearAtRest,DMDcModel):
        return CondensedMPC(model.A,model.B,Q_,R_,P,N,umax,state_constraints)
    return SuccessiveLinearisationMPC(model,Q_,R_,P,N,umax,state_constraints)

def evaluate_loop(model,current=CURRENT,nsteps=NSTEP,keep_trace=False):
    refs=reference(np.arange(nsteps+N_H+1)*DT);z=refs[0].copy();ctl=controller(model)
    d=np.zeros(6);zs=[];us=[];times=[];hs=[];failures=0;nonconverged=0
    for k in range(nsteps):
        start=time.perf_counter()
        try:u,_=ctl.solve(z,refs[k+1:k+N_H+1],d)
        except (MPCSolveError,ValueError,np.linalg.LinAlgError):
            failures+=1;u=us[-1].copy() if us else np.zeros(3)
        qp=ctl.qp if hasattr(ctl,'qp') else ctl
        nonconverged+=int(not getattr(qp,'converged',True))
        times.append(1e3*(time.perf_counter()-start));zs.append(z.copy());us.append(u.copy())
        hs.append(float(model.leverage(z,u)))
        zn=step(z,u,current=current);d=.5*d+.5*(zn-model.predict(z,u));z=zn
        if not np.isfinite(z).all():raise RuntimeError(f'{model.name}: plant diverged')
    zs,us,ms,hs=map(np.array,(zs,us,times,hs));e=zs-refs[:nsteps]
    cross=-np.sin(refs[:nsteps,2])*e[:,0]+np.cos(refs[:nsteps,2])*e[:,1]
    viol=np.maximum(abs(us)-UMAX,0)
    metrics={'cost':float(np.einsum('ti,ij,tj->',e,Q,e)+np.einsum('ti,ij,tj->',us,R,us)),
             'cross_track_rms_m':float(np.sqrt(np.mean(cross**2))),
             'position_rms_m':float(np.sqrt(np.mean(np.sum(e[:,:2]**2,1)))),
             'constraint_violation_count':int(np.any(viol>1e-7,axis=1).sum()),
             'constraint_violation_worst_N':float(viol.max()),
             'solve_median_ms':float(np.median(ms)),'solve_max_ms':float(ms.max()),
             'deadline_misses':int((ms>1000*DT).sum()),'solver_failures':failures,
             'solver_nonconverged':nonconverged,'leverage_median':float(np.median(hs)),
             'leverage_max':float(hs.max()),'leverage_limit':model.hstar,
             'outside_domain_fraction':float(np.mean(hs>model.hstar))}
    if keep_trace:metrics['trace']={'z':zs.tolist(),'u':us.tolist(),'leverage':hs.tolist(),'solve_ms':ms.tolist()}
    return metrics

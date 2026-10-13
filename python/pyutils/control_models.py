"""Part V methods refitted on Part VI's commissioning record.

Models predict discrete next states at dt=.5 s. Absolute x,y are integrators,
not neural-network features: translating a vessel must not change its physics.
Validation chooses hyperparameters; held-out test samples never train a model.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
import numpy as np
from .control import linearize,discretize
from .benchmarks.vessel import VesselParams,state_derivative,step,commissioning_split


LIFT_NAMES=['x','y','psi','u1','v','r','sin(psi)','cos(psi)-1',
            'u1*(cos(psi)-1)','v*sin(psi)','u1*sin(psi)','v*(cos(psi)-1)',
            'abs(u1)*u1','abs(v)*v','abs(r)*r','v*r','u1*r','u1*v',
            'psi*u1','psi*v','psi*r','u1*v*r','abs(u1)*v','abs(v)*u1',
            'sin(psi)*(cos(psi)-1)','u1^3','r^3']


def library(z,rich=False):
    z=np.asarray(z,float); x,y,ps,u,v,r=np.moveaxis(z,-1,0)
    c,s=np.cos(ps)-1,np.sin(ps)
    cols=[x,y,ps,u,v,r,s,c,u*c,v*s,u*s,v*c,abs(u)*u,abs(v)*v,abs(r)*r,v*r,u*r,u*v]
    if rich: cols += [ps*u,ps*v,ps*r,u*v*r,abs(u)*v,abs(v)*u,s*c,u**3,r**3]
    return np.stack(cols,-1)


def input_features(z,u):
    z,u=np.asarray(z),np.asarray(u)
    return np.concatenate([np.sin(z[...,2:3]),np.cos(z[...,2:3]),z[...,3:],u],-1)


class Model:
    name='model'
    training_seconds=0.
    n_parameters=0
    def jacobians_batch(self,Z,U,eps=1e-5):
        Z,U=np.asarray(Z),np.asarray(U)
        A=np.stack([(self.predict(Z+np.eye(Z.shape[-1])[j]*eps,U)-self.predict(Z-np.eye(Z.shape[-1])[j]*eps,U))/(2*eps) for j in range(Z.shape[-1])],-1)
        B=np.stack([(self.predict(Z,U+np.eye(U.shape[-1])[j]*eps)-self.predict(Z,U-np.eye(U.shape[-1])[j]*eps))/(2*eps) for j in range(U.shape[-1])],-1)
        return A,B
    def jacobians(self,z,u):
        A,B=self.jacobians_batch(np.atleast_2d(z),np.atleast_2d(u))
        return A[0],B[0]
    def features(self,z,u):
        return np.concatenate([np.asarray(z)[...,3:],np.asarray(u)],-1)
    def fit_domain(self,Z,U):
        F=self.features(Z,U); self.fm=F.mean(0); self.fs=np.maximum(F.std(0),1e-8)
        X=np.c_[np.ones(len(F)),(F-self.fm)/self.fs]
        self.gram_inv=np.linalg.pinv(X.T@X,rcond=1e-10)
        self.domain_rank=int(np.linalg.matrix_rank(X)); self.hstar=3*self.domain_rank/len(F)
        return self
    def leverage(self,z,u):
        F=self.features(z,u); X=np.concatenate([np.ones(F.shape[:-1]+(1,)),(F-self.fm)/self.fs],-1)
        return np.einsum('...i,ij,...j->...',X,self.gram_inv,X)


class LinearAtRest(Model):
    name='linear'
    def __init__(self,dt=.5,p=None):
        p=p or VesselParams()
        self.A,self.B=discretize(*linearize(lambda z,u:state_derivative(z,u,p),np.zeros(6),np.zeros(3)),dt)
    def predict(self,z,u):return np.asarray(z)@self.A.T+np.asarray(u)@self.B.T
    def jacobians_batch(self,Z,U):
        return np.broadcast_to(self.A,(len(Z),6,6)),np.broadcast_to(self.B,(len(Z),6,3))


class ExactModel(Model):
    name='exact'
    def __init__(self,dt=.5,p=None):self.dt,self.p=dt,p
    def predict(self,z,u):return step(z,u,self.dt,self.p)


class DMDcModel(LinearAtRest):
    name='dmdc'
    def features(self,z,u):return np.concatenate([z,u],-1)
    def fit(self,Z,U,idx):
        start=time.perf_counter()
        X=np.c_[Z[idx],U[idx]]
        G=np.linalg.lstsq(X,Z[idx+1],rcond=1e-10)[0].T
        self.A,self.B=G[:,:6],G[:,6:]; self.n_parameters=G.size
        self.condition_number=float(np.linalg.cond(X));self.training_seconds=time.perf_counter()-start
        return self.fit_domain(Z[idx],U[idx])


def stlsq(X,Y,threshold=.03,iterations=10):
    """Sequential thresholded least squares on unit-RMS columns (Module 21)."""
    scale=np.maximum(np.sqrt((X**2).mean(0)),1e-9); XX=X/scale
    W=np.linalg.lstsq(XX,Y,rcond=1e-9)[0]
    for _ in range(iterations):
        small=abs(W)<threshold
        W[small]=0
        for j in range(Y.shape[1]):
            active=~small[:,j]
            if active.any():W[active,j]=np.linalg.lstsq(XX[:,active],Y[:,j],rcond=1e-9)[0]
    return W/scale[:,None]


class SINDyModel(Model):
    name='sindy'
    def __init__(self,dt=.5,remove_drag=False):
        self.dt=dt; self.indices=[3,4,5]+([] if remove_drag else [12,13,14])+[15,16,17]
    def features(self,z,u):return np.concatenate([library(z,True)[...,self.indices],u],-1)
    def derivative(self,z,u):
        nu=self.features(z,u)@self.W
        ps=z[...,2]; a,b=z[...,3],z[...,4]
        return np.concatenate([np.stack([a*np.cos(ps)-b*np.sin(ps),a*np.sin(ps)+b*np.cos(ps),z[...,5]],-1),nu],-1)
    def predict(self,z,u):
        z=np.asarray(z,float).copy();u=np.asarray(u,float);h=self.dt/5
        for _ in range(5):
            a=self.derivative(z,u);b=self.derivative(z+h*a/2,u);c=self.derivative(z+h*b/2,u);d=self.derivative(z+h*c,u)
            z=z+h*(a+2*b+2*c+d)/6
        return z
    def fit(self,Z,U,train,validation):
        start=time.perf_counter(); mid=(Z[train]+Z[train+1])/2
        X=self.features(mid,U[train]);Y=(Z[train+1,3:]-Z[train,3:])/self.dt
        candidates=[]
        for th in (0.,.001,.003,.01):
            self.W=stlsq(X,Y,th)
            err=np.mean((self.predict(Z[validation],U[validation])-Z[validation+1])**2)
            candidates.append((err,th,self.W.copy()))
        self.validation_mse,self.threshold,self.W=min(candidates,key=lambda a:a[0])
        self.n_parameters=int(np.count_nonzero(self.W));self.training_seconds=time.perf_counter()-start
        return self.fit_domain(Z[train],U[train])


class KoopmanLift(Model):
    name='koopman'
    def __init__(self,indices=None):self.indices=list(range(18)) if indices is None else list(indices)
    def lift(self,z):return library(z,True)[...,self.indices]
    def fit(self,Z,U,idx):
        start=time.perf_counter();X=self.lift(Z[idx]);Y=self.lift(Z[idx+1])
        scale=np.maximum(np.std(np.c_[X,U[idx]],0),1e-8)
        W=np.linalg.lstsq(np.c_[X,U[idx]]/scale,Y,rcond=1e-9)[0].T/scale
        self.A,self.B=W[:,:len(self.indices)],W[:,len(self.indices):]
        self.n_parameters=W.size;self.training_seconds=time.perf_counter()-start
        self.spectral_radius=float(max(abs(np.linalg.eigvals(self.A))))
        return self.fit_domain(Z[idx],U[idx])
    def predict(self,z,u):return (self.lift(z)@self.A.T+np.asarray(u)@self.B.T)[...,:6]
    def rollout_lifted(self,z,U):
        x=self.lift(z);out=[]
        for u in U:x=x@self.A.T+u@self.B.T;out.append(x[...,:6])
        return np.array(out)


class DNNSurrogate(Model):
    name='dnn'
    def __init__(self,seed=0,base=None,width=48):self.seed,self.base,self.width=seed,base,width
    def features(self,z,u):return input_features(z,u)
    def target_base(self,z,u):return np.asarray(z) if self.base is None else self.base.predict(z,u)
    def training_targets(self,Z,U):
        return Z[1:]-self.target_base(Z[:-1],U)
    def fit(self,Z,U,train,validation,epochs=600):
        import torch
        from torch import nn
        torch.set_num_threads(1);torch.manual_seed(self.seed)
        start=time.perf_counter(); X=input_features(Z[:-1],U)
        Y=self.training_targets(Z,U)
        self.xm=X[train].mean(0);self.xs=np.maximum(X[train].std(0),.01)
        self.ym=Y[train].mean(0);self.ys=np.maximum(Y[train].std(0),.001)
        xt=torch.tensor((X-self.xm)/self.xs,dtype=torch.float64);yt=torch.tensor((Y-self.ym)/self.ys,dtype=torch.float64)
        self.net=nn.Sequential(nn.Linear(8,self.width),nn.Tanh(),nn.Linear(self.width,self.width),nn.Tanh(),nn.Linear(self.width,Y.shape[-1])).double()
        opt=torch.optim.Adam(self.net.parameters(),lr=.003,weight_decay=1e-5)
        best=float('inf'); best_state=None
        for epoch in range(epochs):
            opt.zero_grad();loss=((self.net(xt[train])-yt[train])**2).mean();loss.backward();opt.step()
            if epoch%10==0 or epoch==epochs-1:
                with torch.no_grad():val=float(((self.net(xt[validation])-yt[validation])**2).mean())
                if val<best:best=val;best_state=copy.deepcopy(self.net.state_dict());self.best_epoch=epoch
        self.net.load_state_dict(best_state);self.net.eval();self.validation_loss=best
        self.weights=[(layer.weight.detach().numpy().copy(),layer.bias.detach().numpy().copy()) for layer in self.net if isinstance(layer,nn.Linear)]
        self.n_parameters=sum(p.numel() for p in self.net.parameters());self.training_seconds=time.perf_counter()-start
        return self.fit_domain(Z[train],U[train])
    def correction(self,z,u):
        x=(input_features(z,u)-self.xm)/self.xs
        for i,(w,b) in enumerate(self.weights):
            x=x@w.T+b
            if i<len(self.weights)-1:x=np.tanh(x)
        return x*self.ys+self.ym
    def predict(self,z,u):return self.target_base(z,u)+self.correction(z,u)
    def jacobians_batch(self,Z,U):
        import torch
        # Autodiff of the network; chain rule from sin/cos and additive state.
        X=torch.tensor((input_features(Z,U)-self.xm)/self.xs,dtype=torch.float64)
        J=torch.func.vmap(torch.func.jacrev(self.net))(X).detach().numpy()
        J=J*self.ys[None,:,None]/self.xs[None,None,:]
        A=np.zeros((len(Z),6,6));A[:,:,2]=J[:,:,0]*np.cos(Z[:,2,None])-J[:,:,1]*np.sin(Z[:,2,None]);A[:,:,3:]=J[:,:,2:5]
        B=J[:,:,5:]
        if self.base is None:A+=np.eye(6)[None]
        else:
            ab,bb=self.base.jacobians_batch(Z,U);A+=ab;B+=bb
        return A,B


class CoSTAModel(DNNSurrogate):
    name='costa'
    def __init__(self,seed=0,base=None,width=48):super().__init__(seed,base or LinearAtRest(),width)
    def residual_sequence(self,z0,U):
        z=np.asarray(z0,float);ds=[]
        for u in U:ds.append(self.correction(z,u));z=self.predict(z,u)
        return np.array(ds)


def fit_models(Z,U,seed=0,epochs=600):
    split=commissioning_split(len(U));tr,va=split['train'],split['validation']
    models={}
    for model in (LinearAtRest(),DMDcModel(),SINDyModel(),CoSTAModel(seed),DNNSurrogate(seed),KoopmanLift(),ExactModel()):
        if isinstance(model,(DNNSurrogate,SINDyModel)):
            kw={'epochs':epochs} if isinstance(model,DNNSurrogate) else {}
            model.fit(Z,U,tr,va,**kw)
        elif isinstance(model,(DMDcModel,KoopmanLift)):model.fit(Z,U,tr)
        else:model.fit_domain(Z[tr],U[tr])
        models[model.name]=model
    return models


def model_source_digest():
    import hashlib
    from .benchmarks import vessel
    return hashlib.sha256(Path(__file__).read_bytes()+Path(vessel.__file__).read_bytes()).hexdigest()


def save_models(models,path):
    """Local regenerable cache with source provenance; never load untrusted files."""
    import pickle
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'source_sha256':model_source_digest(),'models':models}
    temporary=path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as f:pickle.dump(payload,f)
    temporary.replace(path)


def load_models(path):
    import pickle
    with open(path,'rb') as f:payload=pickle.load(f)
    if payload.get('schema_version')!=1 or payload.get('source_sha256')!=model_source_digest():
        raise ValueError('Stale model cache: rerun mpc_model_box before using these adapters.')
    return payload['models']


class LiftResidual(CoSTAModel):
    """A neural correction in the lifted space, followed by the state readout."""
    name='lift_residual'
    def fit(self,Z,U,train,validation,epochs=600):
        # Reuse the normalised MLP trainer with lifted residual labels.
        self._lift_training=True
        try:return super().fit(Z,U,train,validation,epochs)
        finally:self._lift_training=False
    def training_targets(self,Z,U):
        return self.base.lift(Z[1:])-(self.base.lift(Z[:-1])@self.base.A.T+U@self.base.B.T)
    def predict(self,z,u):return self.base.predict(z,u)+self.correction(z,u)[...,:6]
    def jacobians_batch(self,Z,U):return Model.jacobians_batch(self,Z,U)


class DeepONetHorizon:
    """Surge-only branch--trunk horizon map for SamplingMPC.

    Branch: initial surge speed and five piecewise-constant thrust knots.
    Trunk: the 20 future sample times. Not a one-step 3-DOF vessel model.
    """
    def __init__(self,seed=0,N=20,dt=.5,n_knots=5):
        self.seed,self.N,self.dt,self.n_knots=seed,N,dt,n_knots
    @staticmethod
    def truth(z0,U,dt=.5):
        U=np.atleast_2d(U);z=np.broadcast_to(np.asarray(z0,float), (len(U),)).copy();out=[]
        for j in range(U.shape[1]):
            h=dt/5
            def f(v):return (U[:,j]-10*v-5*abs(v)*v)/55
            for _ in range(5):
                a=f(z);b=f(z+h*a/2);c=f(z+h*b/2);d=f(z+h*c);z+=h*(a+2*b+2*c+d)/6
            out.append(z.copy())
        return np.stack(out,1)
    def fit(self,n=1000,epochs=1200):
        import torch
        from torch import nn
        torch.set_num_threads(1);torch.manual_seed(self.seed);rng=np.random.default_rng(72)
        start=time.perf_counter();self.index=np.minimum(np.arange(self.N)*self.n_knots//self.N,self.n_knots-1)
        z=rng.uniform(-1,1,n);knots=rng.uniform(-30,30,(n,self.n_knots));U=knots[:,self.index]
        X=torch.tensor(np.c_[z,knots/30],dtype=torch.float64)
        T=torch.tensor(np.arange(1,self.N+1)[:,None]/self.N,dtype=torch.float64)
        Y=torch.tensor(self.truth(z,U,self.dt),dtype=torch.float64)
        branch=nn.Sequential(nn.Linear(1+self.n_knots,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh(),nn.Linear(32,24)).double()
        trunk=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,24)).double()
        params=list(branch.parameters())+list(trunk.parameters());opt=torch.optim.Adam(params,lr=.003)
        best=float('inf')
        for epoch in range(epochs):
            pred=branch(X)@trunk(T).T;loss=((pred[:700]-Y[:700])**2).mean()
            if epoch%10==0:
                val=float(((pred[700:850]-Y[700:850])**2).mean().detach())
                if val<best:best=val;self.branch_state=copy.deepcopy(branch.state_dict());self.trunk_state=copy.deepcopy(trunk.state_dict())
            opt.zero_grad();loss.backward();opt.step()
        branch.load_state_dict(self.branch_state);trunk.load_state_dict(self.trunk_state)
        self.branch_weights=[(l.weight.detach().numpy(),l.bias.detach().numpy()) for l in branch if isinstance(l,nn.Linear)]
        with torch.no_grad():self.trunk=trunk(T).numpy()
        self.training_seconds=time.perf_counter()-start;self.n_parameters=sum(p.numel() for p in params)
        self.test_rmse=float(np.sqrt(np.mean((self.predict_horizon(z[850:],U[850:])-Y[850:].numpy())**2)))
        self.rollout_20_rmse=self.test_rmse
        self.one_step_rmse=float(np.sqrt(np.mean((self.predict_horizon(z[850:],U[850:])[:,0]-Y[850:,0].numpy())**2)))
        return self
    def predict_horizon(self,z0,U):
        U=np.atleast_2d(U);knots=U[:,np.arange(self.n_knots)*self.N//self.n_knots]
        z=np.broadcast_to(np.asarray(z0).squeeze(),(len(U),));x=np.c_[z,knots/30]
        for i,(w,b) in enumerate(self.branch_weights):
            x=x@w.T+b
            if i<len(self.branch_weights)-1:x=np.tanh(x)
        return x@self.trunk.T

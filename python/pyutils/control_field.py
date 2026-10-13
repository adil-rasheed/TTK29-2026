"""Canonical heated rod: FTCS, POD, learned Fourier/branch-trunk operators, sensors."""
import time
import numpy as np
from scipy.linalg import qr,solve_discrete_are
from .benchmarks import heat1d
from .control import CondensedMPC


def rod_matrices(nx=128,dt=.02,alpha=.1,c=0.):
    """The exact discrete FTCS map, with zero boundary values eliminated."""
    x=heat1d.grid(nx);dx=x[1]-x[0];n=nx-2
    heaters=np.exp(-((x[:,None]-np.array([.2,.5,.8]))/.06)**2)
    heaters[[0,-1]]=0
    L=alpha*(np.diag(-2*np.ones(n))+np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1))/dx**2
    if c>=0:L+=(-np.eye(n)+np.diag(np.ones(n-1),-1))*c/dx
    else:L+=(-np.eye(n)+np.diag(np.ones(n-1),1))*(-c)/dx
    sub,h=heat1d.steps(dt,nx,alpha)
    if c:sub=max(sub,int(np.ceil(dt*(2*alpha/dx**2+abs(c)/dx)/.8)));h=dt/sub
    K=np.eye(n)+h*L;A=np.linalg.matrix_power(K,sub)
    B=np.linalg.solve(L,(A-np.eye(n))@heaters[1:-1])
    return x,heaters,A,B


def training_data(n=512,seed=42,nx=128):
    x,H,A,B=rod_matrices(nx);rng=np.random.default_rng(seed)
    coefficients=heat1d.sample_family(n,seed);Z=np.array([heat1d.profile(c,x)[1:-1] for c in coefficients])
    U=rng.uniform(-2,2,(n,3));Y=Z@A.T+U@B.T
    return Z,U,Y


class FieldFNO:
    """A learned linear Fourier neural operator on an odd Dirichlet extension.

    Heat flow is linear. A single spectral layer is the appropriate member of
    the FNO family here; two input channels are temperature and heater source.
    Learned Fourier multipliers can be used at a different grid resolution.
    """
    def __init__(self,seed=0,modes=24):self.seed,self.modes=seed,modes
    def fit(self,Z,U,Y,H,epochs=250):
        import torch
        torch.set_num_threads(1);torch.manual_seed(self.seed);start=time.perf_counter()
        n=len(Z);nx=Z.shape[1]+2
        z=np.pad(Z,((0,0),(1,1)));q=U@H.T
        data=np.stack([z,q],1);ext=np.concatenate([data,-data[:,:, -2:0:-1]],2)
        X=torch.fft.rfft(torch.tensor(ext,dtype=torch.float64),dim=-1)
        target=torch.tensor(Y,dtype=torch.float64)
        W=torch.nn.Parameter(.01*torch.randn(2,self.modes,dtype=torch.complex128))
        opt=torch.optim.Adam([W],lr=.05);best=float('inf');tr=slice(0,int(.7*n));va=slice(int(.7*n),int(.85*n))
        for _ in range(epochs):
            F=torch.zeros((n,X.shape[-1]),dtype=torch.complex128)
            F[:,:self.modes]=(X[:,:,:self.modes]*W[None]).sum(1)
            pred=torch.fft.irfft(F,n=ext.shape[-1])[:,1:nx-1]
            loss=((pred[tr]-target[tr])**2).mean();opt.zero_grad();loss.backward();opt.step()
            val=float(((pred[va]-target[va])**2).mean().detach())
            if val<best:best=val;self.W=W.detach().numpy().copy()
        self.training_seconds=time.perf_counter()-start;self.n_parameters=2*self.W.size
        return self
    def predict(self,Z,U,H):
        Z=np.atleast_2d(Z);U=np.atleast_2d(U);nx=Z.shape[1]+2
        data=np.stack([np.pad(Z,((0,0),(1,1))),U@H.T],1)
        ext=np.concatenate([data,-data[:,:,-2:0:-1]],2);X=np.fft.rfft(ext,axis=-1)
        F=np.zeros((len(Z),X.shape[-1]),complex);m=min(self.modes,F.shape[-1])
        F[:,:m]=(X[:,:,:m]*self.W[None,:,:m]).sum(1)
        return np.fft.irfft(F,n=ext.shape[-1],axis=-1)[:,1:nx-1]


class FieldDeepONet:
    """Linear branch--trunk DeepONet with trainable Fourier-feature trunk.

    Sensors of the input field plus three heater settings enter the branch.
    Coordinates enter a sine embedding and trainable trunk matrix. Both are
    learned from FTCS pairs; linear branches reflect this linear PDE.
    """
    def __init__(self,seed=0,rank=16,sensors=32):self.seed,self.rank,self.sensors=seed,rank,sensors
    def fit(self,Z,U,Y,x,epochs=500):
        import torch
        torch.manual_seed(self.seed);start=time.perf_counter();n=len(Z)
        self.sensor_x=np.linspace(0,1,self.sensors+2)[1:-1]
        z=np.pad(Z,((0,0),(1,1)))
        F=np.c_[np.array([np.interp(self.sensor_x,x,a) for a in z]),U]
        X=torch.tensor(F,dtype=torch.float64);T=torch.tensor(np.sin(np.pi*x[1:-1,None]*np.arange(1,self.rank+1)),dtype=torch.float64)
        target=torch.tensor(Y,dtype=torch.float64)
        branch=torch.nn.Parameter(torch.randn(F.shape[1],self.rank,dtype=torch.float64)*.02)
        trunk=torch.nn.Parameter(torch.eye(self.rank,dtype=torch.float64)+torch.randn(self.rank,self.rank,dtype=torch.float64)*.02)
        opt=torch.optim.Adam([branch,trunk],lr=.02);best=float('inf');tr=slice(0,int(.7*n));va=slice(int(.7*n),int(.85*n))
        for _ in range(epochs):
            pred=(X@branch)@(T@trunk).T;loss=((pred[tr]-target[tr])**2).mean()
            opt.zero_grad();loss.backward();opt.step()
            val=float(((pred[va]-target[va])**2).mean().detach())
            if val<best:best=val;self.branch=branch.detach().numpy().copy();self.trunk=trunk.detach().numpy().copy()
        self.training_seconds=time.perf_counter()-start;self.n_parameters=self.branch.size+self.trunk.size
        return self
    def predict(self,Z,U,H=None):
        Z=np.atleast_2d(Z);U=np.atleast_2d(U);x=heat1d.grid(Z.shape[1]+2);z=np.pad(Z,((0,0),(1,1)))
        F=np.c_[np.array([np.interp(self.sensor_x,x,a) for a in z]),U]
        T=np.sin(np.pi*x[1:-1,None]*np.arange(1,self.rank+1))
        return (F@self.branch)@(T@self.trunk).T


def identified_operator(model,n,H):
    """Probe a learned linear operator, not the PDE, for offline condensing."""
    A=model.predict(np.eye(n),np.zeros((n,3)),H).T
    B=model.predict(np.zeros((3,n)),np.eye(3),H).T
    return A,B


def field_mission(A,B,truth_A,truth_B,basis=None,sensors=None,steps=80):
    n=len(truth_A);r=len(A);Q=np.eye(r)/n;R=np.eye(3)*.002
    P=solve_discrete_are(A,B,Q,R);ctl=CondensedMPC(A,B,Q,R,P,20,[2,2,2])
    x=heat1d.grid(n+2)[1:-1];z=np.sin(np.pi*x)+.2*np.sin(8*np.pi*x)
    refs=.3*np.sin(np.pi*x);us=[];zs=[];timings=[]
    Phi=np.eye(n) if basis is None else basis
    estimate=np.eye(n) if sensors is None else Phi@np.linalg.pinv(Phi[sensors])
    for _ in range(steps):
        start=time.perf_counter();zhat=z if sensors is None else estimate@z[sensors]
        u,_=ctl.solve(Phi.T@zhat,Phi.T@refs);timings.append(1e3*(time.perf_counter()-start))
        zs.append(z.copy());us.append(u);z=truth_A@z+truth_B@u
    Z,U,ms=np.array(zs),np.array(us),np.array(timings)
    cost=float(((Z-refs)**2).sum()/n+.002*(U**2).sum())
    return {'cost':cost,'solve_median_ms':float(np.median(ms)),'solve_max_ms':float(ms.max()),
            'deadline_misses':int((ms>20).sum()),'budget_ms':20,'steps':steps,
            'constraint_violation_count':int(np.any(abs(U)>2+1e-7,1).sum()),
            'constraint_violation_worst':float(np.maximum(abs(U)-2,0).max())}


def field_rollout_error(ar,br,A,B,Z,U,basis=None):
    V=np.eye(len(A)) if basis is None else basis
    truth=Z.copy();state=Z@V;errors=[]
    for _ in range(20):
        truth=truth@A.T+U@B.T;state=state@ar.T+U@br.T
        errors.append(state@V.T-truth)
    return float(np.sqrt(np.mean(np.array(errors)**2)))


def field_experiment():
    start=time.perf_counter();x,H,A,B=rod_matrices();Z,U,Y=training_data();n=len(A);test=slice(435,None)
    # Matrix representation is checked against the canonical explicit solver.
    truth=heat1d.ftcs(np.r_[0,Z[0],0],t_end=.02,q=H@U[0])[1:-1]
    np.testing.assert_allclose(A@Z[0]+B@U[0],truth,atol=1e-12)
    Phi,sv,_=np.linalg.svd(np.vstack([Z[:358],Y[:358]]).T,full_matrices=False)
    rows=[{'model':'FTCS','seed':0,'dimension':n,'training_seconds':0.,'one_step_rmse':0.,'rollout_20_rmse':0.,'parameters':0,**field_mission(A,B,A,B)}]
    for r in (2,4,8,12):
        V=Phi[:,:r];ar,br=V.T@A@V,V.T@B
        rows.append({'model':f'POD {r}','seed':0,'dimension':r,'training_seconds':0.,'parameters':int(V.size),
                     'rollout_20_rmse':field_rollout_error(ar,br,A,B,Z[test],U[test],V),
                     'one_step_rmse':float(np.sqrt(np.mean(((Z[test]@V@ar.T+U[test]@br.T)@V.T-Y[test])**2))),
                     **field_mission(ar,br,A,B,V)})
    for seed in (0,1,2):
        for model,name in [(FieldFNO(seed).fit(Z,U,Y,H),'FNO'),(FieldDeepONet(seed).fit(Z,U,Y,x),'DeepONet')]:
            ar,br=identified_operator(model,n,H)
            err=float(np.sqrt(np.mean((model.predict(Z[test],U[test],H)-Y[test])**2)))
            x2,H2,A2,B2=rod_matrices(64);Z2,U2,Y2=training_data(80,72,64)
            transfer=float(np.sqrt(np.mean((model.predict(Z2,U2,H2)-Y2)**2)))
            rows.append({'model':name,'seed':seed,'dimension':n,'training_seconds':model.training_seconds,
                         'parameters':model.n_parameters,'one_step_rmse':err,'resolution_128_to_64_rmse':transfer,
                         'rollout_20_rmse':field_rollout_error(ar,br,A,B,Z[test],U[test]),
                         **field_mission(ar,br,A,B)})
    V=Phi[:,:8];_,_,piv=qr(V.T,pivoting=True)
    sensors=[]
    for count in (8,12,16):
        idx=piv[:count];row={'sensors':count,'indices':idx.tolist(),**field_mission(V.T@A@V,V.T@B,A,B,V,idx)}
        sensors.append(row)
    full=rows[0]['cost']
    for row in rows:row['cost_penalty_percent']=100*(row['cost']/full-1)
    return {'rows':rows,'sparse_sensors':sensors,'elapsed_seconds':time.perf_counter()-start,
            'protocol':{'plant':'canonical heat1d.ftcs','alpha':.1,'advection_c':0.,'grid':128,'dt':.02,
                        'heaters':[.2,.5,.8],'gaussian_width':.06,'horizon':20,'input_bounds':[-2,2],
                        'split':'first 358 train, next 77 validation, last 77 test; independent profiles',
                        'cost':'sum mean field squared tracking error + .002 sum heater input squared',
                        'operators':'Part V architectures refitted with heater forcing; linear PDE operators condensed offline'}}

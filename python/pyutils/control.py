"""Shared prediction, condensed MPC and loop logging for Part VI.

The notebooks derive the controller; this implementation gives each experiment
one numerical solver to reuse when only the model box is changed. The constant
matrix path preserves the original BVLS/SLSQP arithmetic. Nonlinear adapters
use the same QP with time-varying affine prediction matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["linearize", "discretize", "ClosedLoopLog", "simulate_closed_loop"]


def linearize(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    z0: np.ndarray,
    u0: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians of ``f(z, u)`` about an operating point.

    Returns the continuous-time pair ``(A, B)``. Central differences rather
    than forward: the error is O(eps^2), which matters because these matrices
    are then fed to a Riccati solve that squares things.
    """
    z0 = np.asarray(z0, float)
    u0 = np.asarray(u0, float)
    n, m = z0.size, u0.size

    A = np.zeros((n, n))
    for j in range(n):
        dz = np.zeros(n)
        dz[j] = eps
        A[:, j] = (f(z0 + dz, u0) - f(z0 - dz, u0)) / (2 * eps)

    B = np.zeros((n, m))
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        B[:, j] = (f(z0, u0 + du) - f(z0, u0 - du)) / (2 * eps)

    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation via the matrix exponential.

    Uses the block-matrix trick: exp([[A, B], [0, 0]] dt) has Ad in its top-left
    block and Bd in its top-right. Exact for a ZOH input, unlike Euler, which
    quietly biases the identified model at large dt.
    """
    from scipy.linalg import expm

    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


@dataclass
class ClosedLoopLog:
    """What a closed-loop run produced. Arrays are (steps, dim)."""

    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    z: np.ndarray = field(default_factory=lambda: np.empty(0))
    u: np.ndarray = field(default_factory=lambda: np.empty(0))
    solve_ms: np.ndarray = field(default_factory=lambda: np.empty(0))

    def cost(self, Q: np.ndarray, R: np.ndarray, z_ref: np.ndarray | None = None) -> float:
        """Accumulated quadratic stage cost -- the number controllers compete on."""
        e = self.z if z_ref is None else self.z - z_ref
        return float(np.einsum("ti,ij,tj->", e, Q, e) + np.einsum("ti,ij,tj->", self.u, R, self.u))


def simulate_closed_loop(
    step: Callable[[np.ndarray, np.ndarray], np.ndarray],
    policy: Callable[[int, np.ndarray], np.ndarray],
    z0: np.ndarray,
    n_steps: int,
    dt: float,
) -> ClosedLoopLog:
    """Run ``policy`` against ``step`` and record the trajectory.

    ``step(z, u) -> z_next`` is the plant for one sample; ``policy(k, z) -> u``
    is the controller. Solve time is recorded per step because "does it fit in
    the sample interval" is a real-time claim these lectures have to make with
    a number rather than a hope.
    """
    import time

    z = np.asarray(z0, float)
    zs, us, ms = [], [], []
    for k in range(n_steps):
        t0 = time.perf_counter()
        u = np.atleast_1d(np.asarray(policy(k, z), float))
        ms.append((time.perf_counter() - t0) * 1e3)
        zs.append(z.copy())
        us.append(u.copy())
        z = np.asarray(step(z, u), float)

    return ClosedLoopLog(
        t=np.arange(n_steps) * dt,
        z=np.array(zs),
        u=np.array(us),
        solve_ms=np.array(ms),
    )


class MPCSolveError(RuntimeError):
    """The optimiser did not return a feasible plan; callers must handle it."""


class CondensedMPC:
    """Box/state constrained quadratic tracking with optional affine forcing.

    ``A, B`` may be constant or arrays with a leading horizon dimension.
    ``state_constraints=(C, lower, upper)`` imposes bounds on C z at every
    predicted step; bounds may have shape (N, C.shape[0]). Infinity is allowed.
    ``d_seq`` is either a constant n-vector or (N,n) discrete additive offsets.
    ``solve`` returns (first input, elapsed milliseconds), retaining ``plan``
    and ``predicted``. Timing includes the optimiser and right-hand side.
    """
    def __init__(self, A, B, Q, R, P, N, umax, state_constraints=None,
                 solver='bvls'):
        from scipy.linalg import block_diag, cholesky
        self.N = int(N)
        if self.N < 1 or solver not in ('bvls', 'slsqp'):
            raise ValueError('positive horizon and bvls/slsqp solver required')
        self.Q, self.R, self.P = map(lambda x: np.asarray(x, float), (Q, R, P))
        self.n, self.m = self.Q.shape[0], self.R.shape[0]
        self.umax = np.broadcast_to(np.asarray(umax, float), (self.m,)).copy()
        if np.any(self.umax <= 0):
            raise ValueError('umax must be positive')
        self.LO, self.HI = np.tile(-self.umax, N), np.tile(self.umax, N)
        self.Qbar = block_diag(*([self.Q] * (N - 1) + [self.P]))
        self.Rbar = block_diag(*([self.R] * N))
        def cost_factor(W):
            try:
                return cholesky(W)  # preserve legacy positive-definite arithmetic
            except np.linalg.LinAlgError:
                values, vectors = np.linalg.eigh(W)
                if values.min() < -1e-10:
                    raise ValueError('cost matrices must be positive semidefinite')
                return np.diag(np.sqrt(np.maximum(values, 0))) @ vectors.T
        self.LQ, self.LR = cost_factor(self.Qbar), cost_factor(self.Rbar)
        self.state_constraints, self.solver = state_constraints, solver
        self.plan = np.zeros((N, self.m))
        self.warm_start = np.zeros(N * self.m)
        self.update_dynamics(A, B)

    def update_dynamics(self, A, B):
        n, m, N = self.n, self.m, self.N
        A, B = np.asarray(A, float), np.asarray(B, float)
        self.A, self.B = A, B
        self.Su = np.zeros((N*n, N*m))
        self.Sw = np.zeros((N*n, N*n))
        # Preserve the original notebooks' operation order for constant models.
        if A.ndim == B.ndim == 2:
            pw = [np.eye(n)]
            for _ in range(N):
                pw.append(pw[-1] @ A)
            self.Sz = np.vstack([pw[i+1] for i in range(N)])
            self.Sd = np.zeros((N*n, n))
            acc = np.zeros((n, n))
            for i in range(N):
                acc = acc + pw[i]
                self.Sd[i*n:(i+1)*n] = acc
                for j in range(i+1):
                    self.Su[i*n:(i+1)*n, j*m:(j+1)*m] = pw[i-j] @ B
                    self.Sw[i*n:(i+1)*n, j*n:(j+1)*n] = pw[i-j]
        else:
            As = np.broadcast_to(A, (N,n,n)); Bs = np.broadcast_to(B, (N,n,m))
            self.Sz = np.zeros((N*n,n))
            az = np.eye(n)
            for i in range(N):
                az = As[i] @ az
                self.Sz[i*n:(i+1)*n] = az
                self.Su[i*n:(i+1)*n, i*m:(i+1)*m] = Bs[i]
                self.Sw[i*n:(i+1)*n, i*n:(i+1)*n] = np.eye(n)
                if i:
                    self.Su[i*n:(i+1)*n, :i*m] = As[i] @ self.Su[(i-1)*n:i*n, :i*m]
                    self.Sw[i*n:(i+1)*n, :i*n] = As[i] @ self.Sw[(i-1)*n:i*n, :i*n]
            self.Sd = self.Sw @ np.tile(np.eye(n), (N,1))
        self.G = np.vstack([self.LQ @ self.Su, self.LR])
        self.HESS = self.Su.T @ self.Qbar @ self.Su + self.Rbar

    def solve(self, z0, zref, d_seq=None):
        import time
        from scipy.optimize import lsq_linear, minimize
        t0 = time.perf_counter()
        ref = np.asarray(zref, float)
        if ref.shape == (self.n,):
            ref = np.tile(ref, (self.N, 1))
        ref = ref.reshape(self.N * self.n)
        free = self.Sz @ np.asarray(z0, float)
        if d_seq is not None:
            d = np.asarray(d_seq, float)
            free = free + (self.Sd @ d if d.shape == (self.n,) else self.Sw @ d.reshape(-1))
        if not np.isfinite(free).all():
            raise MPCSolveError('non-finite prediction')
        if self.state_constraints is None and self.solver == 'bvls':
            rhs = np.concatenate([-self.LQ @ (free-ref), np.zeros(self.N*self.m)])
            result = lsq_linear(self.G, rhs, bounds=(self.LO,self.HI),
                                method='bvls', lsq_solver='exact')
        else:
            g = self.Su.T @ self.Qbar @ (free-ref)
            cons = ()
            if self.state_constraints is not None:
                C, lower, upper = self.state_constraints
                C = np.atleast_2d(C)
                CC = np.kron(np.eye(self.N), C)
                lo = np.broadcast_to(lower, (self.N,len(C))).ravel()
                hi = np.broadcast_to(upper, (self.N,len(C))).ravel()
                cf, cu = CC @ free, CC @ self.Su
                ih, il = np.isfinite(hi), np.isfinite(lo)
                b = np.concatenate([(hi-cf)[ih], (cf-lo)[il]])
                D = np.vstack([cu[ih], -cu[il]])
                def residual(U):
                    cuU = CC @ (self.Su @ U)
                    return np.concatenate([(hi-cf-cuU)[ih], (cf-lo+cuU)[il]])
                cons = ({'type':'ineq', 'fun':residual, 'jac':lambda U: -D},)
            result = minimize(lambda U: .5*U@self.HESS@U+g@U, self.warm_start,
                              jac=lambda U: self.HESS@U+g,
                              bounds=list(zip(self.LO,self.HI)), constraints=cons,
                              method='SLSQP', options={'maxiter':80,'ftol':1e-9})
        self.last_result = result
        # A feasible iterate at the iteration limit is usable but not certified
        # optimal; expose termination status rather than silently claiming it.
        self.converged = bool(result.success)
        U = result.x
        feasible = np.isfinite(U).all() and np.max(np.abs(U)-self.HI) <= 1e-7
        if self.state_constraints is not None:
            feasible = feasible and np.min(b-D@U) >= -1e-6
        if not feasible:
            raise MPCSolveError(str(result.message))
        self.plan = U.reshape(self.N,self.m).copy()
        self.predicted = (free+self.Su@U).reshape(self.N,self.n)
        self.warm_start = np.concatenate([U[self.m:],np.zeros(self.m)])
        return self.plan[0].copy(), 1e3*(time.perf_counter()-t0)


class SuccessiveLinearisationMPC:
    """One real-time SQP iteration along the shifted previous input plan.

    The affine term f(z,u)-Az-Bu is essential: relinearisation without it
    silently changes the model. The same condensed QP and costs are retained.
    """
    def __init__(self, model, Q, R, P, N, umax, state_constraints=None):
        self.model, self.N = model, N
        self.U = np.zeros((N,len(R)))
        A,B = model.jacobians(np.zeros(len(Q)),np.zeros(len(R)))
        self.qp = CondensedMPC(A,B,Q,R,P,N,umax,state_constraints)

    def solve(self,z0,zref,d_seq=None):
        import time
        start=time.perf_counter()
        z=np.asarray(z0,float).copy(); Z=[]
        for u in self.U:
            Z.append(z); z=self.model.predict(z,u)
        Z=np.array(Z)
        if hasattr(self.model,'jacobians_batch'):
            As,Bs=self.model.jacobians_batch(Z,self.U)
        else:
            pairs=[self.model.jacobians(z,u) for z,u in zip(Z,self.U)]
            As,Bs=map(np.array,zip(*pairs))
        Y=self.model.predict(Z,self.U)
        ds=Y-np.einsum('tij,tj->ti',As,Z)-np.einsum('tij,tj->ti',Bs,self.U)
        if d_seq is not None: ds=ds+np.asarray(d_seq)
        self.qp.update_dynamics(As,Bs)
        u,_=self.qp.solve(z0,zref,ds)
        self.plan=self.qp.plan.copy()
        self.U=np.vstack([self.plan[1:],self.plan[-1]])
        return u,1e3*(time.perf_counter()-start)


class SamplingMPC:
    """CEM over piecewise-constant thrust histories, using a batched horizon map."""
    def __init__(self,horizon_model,n_candidates=256,N=20,umax=30.,Q=20.,R=.002,
                 n_knots=5,iterations=3,seed=0):
        self.model=horizon_model; self.n_candidates=n_candidates; self.N=N
        self.umax=umax; self.Q=Q; self.R=R; self.n_knots=n_knots; self.iterations=iterations
        self.rng=np.random.default_rng(seed); self.mean=np.zeros(n_knots)

    def solve(self,z0,zref,d_seq=None):
        import time
        start=time.perf_counter(); mean=self.mean.copy(); std=np.full(self.n_knots,self.umax/2)
        index=np.minimum(np.arange(self.N)*self.n_knots//self.N,self.n_knots-1)
        for _ in range(self.iterations):
            knots=np.clip(self.rng.normal(mean,std,(self.n_candidates,self.n_knots)),-self.umax,self.umax)
            U=knots[:,index]; pred=self.model.predict_horizon(z0,U)
            if d_seq is not None: pred=pred+np.asarray(d_seq)
            cost=self.Q*((pred-np.asarray(zref))**2).sum(1)+self.R*(U**2).sum(1)
            elite=knots[np.argsort(cost)[:max(2,self.n_candidates//10)]]
            mean,std=elite.mean(0),np.maximum(elite.std(0),.05)
        best=int(np.argmin(cost)); self.plan=U[best]; self.mean=mean
        return np.atleast_1d(self.plan[0]),1e3*(time.perf_counter()-start)

__all__ += ['CondensedMPC','SuccessiveLinearisationMPC','SamplingMPC','MPCSolveError']

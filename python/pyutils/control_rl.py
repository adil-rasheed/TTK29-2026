"""The Module 29 PPO learner, with an injectable training twin.

Algorithm, architecture, optimiser and rollout settings match rl_vs_mpc.
The twin is the only experimental factor changed by rl_learned_twins.
"""
import numpy as np
import torch
from torch import nn
from .dl import set_dl_defaults
from .control_experiments import reference,DT,NSTEP,UMAX,Q,R
U_MAX=UMAX
DEV=torch.device('cpu')
N_OBS,N_ACT=9,3
N_ENV,HOR=64,120
GAMMA,LAM,CLIP,R_SCALE=.99,.95,.2,.05
Z_REF=reference(np.arange(NSTEP+22)*DT)
wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi

def stage_cost(z, zr, u):
    """The Module 27 cost, per sample, with the heading error wrapped."""
    e = z - zr
    e = np.concatenate([e[:, :2], wrap(e[:, 2:3]), e[:, 3:]], 1)
    return np.einsum("ti,ij,tj->t", e, Q, e) + np.einsum("ti,ij,tj->t", u, R, u)

def observe(z, zr):
    c, s = np.cos(z[:, 2]), np.sin(z[:, 2])
    ex, ey = zr[:, 0] - z[:, 0], zr[:, 1] - z[:, 1]
    return np.stack([0.5 * (c*ex + s*ey), 0.5 * (-s*ex + c*ey),
                     wrap(zr[:, 2] - z[:, 2]),
                     2 * (zr[:, 3] - z[:, 3]), 2 * (zr[:, 4] - z[:, 4]),
                     2 * (zr[:, 5] - z[:, 5]),
                     2 * zr[:, 3], 2 * zr[:, 4], 2 * zr[:, 5]], 1).astype(np.float32)

def env_reset(n, rng):
    """Start anywhere on the transit, displaced off it -- the agent must be able
    to recover, not just to stay."""
    k0 = rng.integers(0, NSTEP - HOR, size=n)
    z = Z_REF[k0].copy()
    z[:, :2] += rng.normal(0, 1.2, (n, 2))
    z[:, 2] += rng.normal(0, 0.15, n)
    z[:, 3:] += rng.normal(0, 0.08, (n, 3))
    return z, k0

def mlp(nin, nout, h=64):
    return nn.Sequential(nn.Linear(nin, h), nn.Tanh(),
                         nn.Linear(h, h), nn.Tanh(), nn.Linear(h, nout))

class Agent:
    def __init__(self, seed, twin_step):
        self.twin_step = twin_step
        set_dl_defaults(seed=seed)
        self.actor = mlp(N_OBS, N_ACT).to(DEV)
        self.critic = mlp(N_OBS, 1).to(DEV)
        self.log_std = nn.Parameter(torch.full((N_ACT,), -0.7))
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters())
            + [self.log_std], lr=3e-4)
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        """Deterministic control law: the policy mean, scaled to thrust."""
        with torch.no_grad():
            a = torch.tanh(self.actor(torch.from_numpy(obs))).numpy()
        return np.clip(a, -1.0, 1.0) * U_MAX

    def rollout(self):
        z, k = env_reset(N_ENV, self.rng)
        O = np.zeros((HOR, N_ENV, N_OBS), np.float32)
        A = np.zeros((HOR, N_ENV, N_ACT), np.float32)
        LP = np.zeros((HOR, N_ENV), np.float32)
        RW = np.zeros((HOR, N_ENV), np.float32)
        V = np.zeros((HOR + 1, N_ENV), np.float32)
        for t in range(HOR):
            o = observe(z, Z_REF[k])
            ot = torch.from_numpy(o)
            with torch.no_grad():
                mu = torch.tanh(self.actor(ot))
                V[t] = self.critic(ot).squeeze(-1).numpy()
                std = self.log_std.exp()
                a = mu + std * torch.randn_like(mu)
                lp = (-0.5 * ((a - mu) / std)**2 - self.log_std - 0.9189385).sum(-1)
            an = a.numpy()
            u = np.clip(an, -1.0, 1.0) * U_MAX
            z_next = self.twin_step(z, u)
            RW[t] = -R_SCALE * stage_cost(z_next, Z_REF[k + 1], u)
            O[t], A[t], LP[t] = o, an, lp.numpy()
            z, k = z_next, k + 1
        with torch.no_grad():
            V[HOR] = self.critic(torch.from_numpy(observe(z, Z_REF[k]))).squeeze(-1).numpy()
        adv = np.zeros((HOR, N_ENV), np.float32); g = 0.0
        for t in reversed(range(HOR)):          # generalised advantage estimation
            g = RW[t] + GAMMA * V[t + 1] - V[t] + GAMMA * LAM * g
            adv[t] = g
        ret = adv + V[:HOR]
        flat = lambda x, d: torch.from_numpy(x.reshape(-1, d) if d else x.reshape(-1))
        return flat(O, N_OBS), flat(A, N_ACT), flat(LP, 0), flat(adv, 0), flat(ret, 0)

    def update(self, O, A, LP0, ADV, RET, epochs=4, mb=768):
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        for _ in range(epochs):
            idx = torch.randperm(O.shape[0])
            for i in range(0, O.shape[0], mb):
                j = idx[i:i + mb]
                mu = torch.tanh(self.actor(O[j])); std = self.log_std.exp()
                lp = (-0.5 * ((A[j] - mu) / std)**2 - self.log_std - 0.9189385).sum(-1)
                ratio = (lp - LP0[j]).exp()
                l_pi = -torch.min(ratio * ADV[j],
                                  ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[j]).mean()
                l_v = ((self.critic(O[j]).squeeze(-1) - RET[j])**2).mean()
                loss = l_pi + 0.5 * l_v - 1e-3 * self.log_std.sum()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                self.opt.step()

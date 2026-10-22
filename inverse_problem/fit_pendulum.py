"""
Fit the windy pendulum's port-Hamiltonian parameters from trajectory data.

Six positive scalars -- I, gamma, b_base, b_amp, sigma_gust, sigma_turb -- by
autodiff through the SAME stochastic-Heun map the data came from.  Parameters
rather than networks, deliberately: on this plant M is a constant and V is one
cosine, so a six-scalar fit tells you everything a network could and makes the
recovery numbers directly checkable against truth.  Swapping in the M^-1/V/D
nets changes this file only -- IDAPBC consumes a Params either way.

WHAT THE LOSS LOOKS LIKE, AND WHY.  One step from an observed state, never a
rollout: the model cannot reproduce the realized wind path, so a multi-step
objective would measure divergence instead of parameter error.  The residual
then splits along the rank-one diffusion --

    theta   noise-free, carries I and (through the corrector) most of gamma
    p       carries all the wind; its SPREAD identifies sigma, so a Gaussian
            likelihood, not least squares

-- and the two terms need balancing, because the likelihood is O(1) while the
squared error is O(1e-3).  At equal weight the optimizer spends itself on sigma
and damping comes back at essentially zero; `det_weight` is the fix, with the
same value that worked on the cart-pole.

WHAT ACTUALLY MATTERS HERE.  I and gamma come back to ~0.2%; damping to 20-30%.
That split is not a defect to tune away -- damping sits in the noisy channel and
contributes about six times less than the wind per step, so 20-30% is near what
the data supports.  It also does not matter: IDA-PBC CANCELS D and replaces it
with injected k_d, so a 0.09 absolute error perturbs the closed loop by ~4%.
The terms the controller is sensitive to are exactly the ones that are easy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import pendulum as pend
from .pendulum import Params

PARAM_INIT = (0.5, 5.0, 0.1, 0.1, 0.05, 0.05)


def make_dataset(p: Params, n_traj=30, n_steps=400, dt=0.02, seed=0, amplitude=8.0):
    """Short trajectories from spread-out initial conditions.

    Several rather than one long run: the fit needs the pendulum seen at many
    angles AND many speeds, since damping is distinguished from gravity only by
    its velocity dependence.
    """
    rng = np.random.default_rng(seed)
    Z, U, ZN = [], [], []
    for j in range(n_traj):
        z0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-4.0, 4.0)])
        tr = pend.generate(p, n_steps=n_steps, dt=dt, z0=z0, amplitude=amplitude,
                           seed=seed * 1000 + j)
        Z.append(tr.z[:-1]); ZN.append(tr.z[1:]); U.append(tr.u)
    t = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float64)
    return t(Z), t(U), t(ZN)


def _predict(z, u, q, dt, dW=None):
    """Batched stochastic-Heun step in torch.  Mirrors pendulum.heun_step,
    including the asymmetric theta predictor the env uses."""
    I, gamma, b_base, b_amp, sg, st = q
    th, mom = z[:, 0], z[:, 1]
    thd = mom / I
    D = b_base + b_amp * torch.abs(torch.sin(th))
    pd = -gamma * torch.sin(th) + u - D * thd
    s0 = sg + st * torch.abs(thd)
    th_t = th + thd * dt
    p_t = mom + pd * dt + (0.0 if dW is None else s0 * dW)
    thd_t = p_t / I
    D_t = b_base + b_amp * torch.abs(torch.sin(th_t))
    pd_t = -gamma * torch.sin(th_t) + u - D_t * thd_t
    s1 = sg + st * torch.abs(thd_t)
    nxt = torch.stack([
        th + 0.5 * (thd + thd_t) * dt,
        mom + 0.5 * (pd + pd_t) * dt + (0.0 if dW is None else 0.5 * (s0 + s1) * dW),
    ], dim=1)
    return nxt, s0


@dataclass
class FitReport:
    params: Params
    true: Params | None
    steps: int
    n_transitions: int

    def table(self) -> str:
        rows = ["  param        learned      true     rel.err"]
        tv = self.true.to_vector() if self.true is not None else [None] * 6
        for n, a, b in zip(pend.PARAM_NAMES, self.params.to_vector(), tv):
            err = "" if b is None else f"{abs(a - b) / abs(b) * 100:7.2f}%"
            tb = "" if b is None else f"{b:9.4f}"
            rows.append(f"  {n:11s} {a:9.4f} {tb}  {err}")
        return "\n".join(rows)


def fit(z, u, z_next, dt=0.02, steps=4000, batch=1024, lr=0.05,
        det_weight=100.0, seed=0, true: Params | None = None) -> FitReport:
    torch.manual_seed(seed)
    sp = torch.nn.functional.softplus
    raw = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        raw.copy_(torch.log(torch.expm1(torch.tensor(PARAM_INIT, dtype=torch.float64))))

    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scale = (z_next - z).std(dim=0).clamp_min(1e-12)
    g = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        i = torch.randint(0, len(z), (batch,), generator=g)
        pred, s0 = _predict(z[i], u[i], sp(raw), dt)
        r = z_next[i] - pred
        det = ((r[:, 0] / scale[0]) ** 2).mean()
        var = (s0.clamp_min(1e-8) ** 2) * dt
        nll = 0.5 * ((r[:, 1] ** 2) / var + torch.log(var)).mean()
        opt.zero_grad(set_to_none=True)
        (det_weight * det + nll).backward()
        opt.step()
        sched.step()

    return FitReport(params=Params(*sp(raw).detach().numpy()), true=true,
                     steps=steps, n_transitions=len(z))


def fit_from_params(true: Params, dt=0.02, n_traj=30, n_steps=400, seed=0,
                    **fit_kwargs) -> FitReport:
    """Convenience: generate data at `true` and fit it back."""
    z, u, zn = make_dataset(true, n_traj=n_traj, n_steps=n_steps, dt=dt, seed=seed)
    return fit(z, u, zn, dt=dt, seed=seed, true=true, **fit_kwargs)

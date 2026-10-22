"""
Fit the learned port-Hamiltonian model to simulated cart-pole data.

    .venv/bin/python -m inverse_problem.train --steps 3000

THE OBJECTIVE IS PER-STEP, NOT PER-TRAJECTORY.  Each prediction starts from an
OBSERVED z_k and takes one Heun step, so error never compounds.  Rolling the
model out and matching whole trajectories would be wrong here for a specific
reason: the model cannot reproduce the realized wind path, so even at the true
parameters a rollout diverges from the data, and the loss would end up measuring
divergence rather than parameter error.

THE RESIDUAL SPLITS ALONG THE NOISE STRUCTURE.  The diffusion is rank one in a
four-dimensional state -- wind hits p_theta and nothing else.  So:

    x, theta, p_x   noise-free IN THE SDE.  Residuals carry almost all of the
                    information about M, V and D.  But note they are NOT noise
                    free in the DISCRETE map: Heun's corrector evaluates the
                    drift at a predictor that already contains the p_theta wind
                    kick, so the noise leaks into theta and x within the same
                    step, at O(sigma * dt^{3/2}).  Measured here that is ~2.6%
                    of the theta increment scale, and the theta residual sits at
                    ~6.4% no matter how the loss is weighted -- a floor this
                    objective cannot push through.  The EKF marginal likelihood
                    handles it properly, because its process-noise covariance
                    B_k B_k^T is obtained by differentiating the integrator step
                    with respect to the noise and therefore sees the leakage.
    p_theta         residual IS the wind, ~ sigma(z_k) dW_k.  Driving it to zero
                    would be fitting noise; its SPREAD identifies sigma, so this
                    channel gets a Gaussian likelihood with learned variance.

KNOWN LIMITATION.  A per-step likelihood sees only the combined variance
sigma^2 dt + 2 sigma_obs^2, so it cannot separate process noise from measurement
noise (LieSPHGP sec 3.4 makes this point precisely).  With full-state noise-free
sensing, which is the default here, that does not bite.  Turn observation noise
on and sigma becomes biased -- at which point the objective has to become the
EKF marginal likelihood.  The model and the data pipeline do not change when it
does; only this file does.

DATA COVERAGE IS A FIRST-CLASS CHOICE.  LieSPHGP's neural baseline predicted
well in open loop and still broke its controller, because no training state came
within 10 degrees of upright and the learned physics was wrong exactly there.
`make_dataset` therefore spreads initial angles deliberately, including near
upright.  --theta-spread exists so that failure can be reproduced on purpose.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from environment.cartpole import CartPoleEnv
from inverse_problem import data as D
from inverse_problem.dynamics import Params
from inverse_problem.model import PortHamiltonianModel, TrueModel


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_dataset(p: Params, n_traj=24, n_steps=400, dt=0.02, seed=0,
                 theta_spread=np.pi, obs_noise=0.0):
    """Several short trajectories from spread-out initial conditions.

    Several rather than one long run, for a structural reason: M and V both feed
    dH/dtheta and are distinguished ONLY by the p-dependence (the mass term is
    quadratic in p, the potential term has no p at all).  A single low-energy
    trajectory therefore leaves them trading off along a flat direction.  Varying
    the initial momenta is what separates them.
    """
    rng = np.random.default_rng(seed)
    Z, U, Znext = [], [], []
    for j in range(n_traj):
        theta0 = rng.uniform(-theta_spread, theta_spread)
        z0 = D.initial_state(
            p, x=rng.uniform(-0.3, 0.3), theta=theta0,
            xdot=rng.uniform(-0.8, 0.8), thetadot=rng.uniform(-2.5, 2.5))
        traj = D.generate(p, n_steps=n_steps, dt=dt, z0=z0,
                          input_design="random_steps", sensing="full",
                          obs_noise_std=obs_noise, seed=seed * 1000 + j,
                          input_kwargs=dict(amplitude=6.0, hold_seconds=0.2))
        Z.append(traj.obs[:-1])
        Znext.append(traj.obs[1:])
        U.append(traj.u)
    t = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float64)
    return t(Z), t(U), t(Znext)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def step_loss(model, z, u, z_next, dt, scale, det_weight=100.0):
    """Per-step loss.  Returns (total, det_mse, wind_nll).

    `det_weight` is NOT cosmetic.  The wind NLL is O(1) while the squared error
    on the quiet channels is O(1e-3), so at equal weight the optimizer spends
    itself fitting sigma and leaves a p_x residual the same size as the entire
    damping signal -- which makes D unidentifiable.  Measured, 3000 steps:

        det_weight      M^-1 err    D err    D(0) diag        p_x residual
             1           0.0133     0.751    [0.0003, 0.0008]    3.3%
           100           0.0053     0.118    [0.0775, 0.0140]    0.09%
         10000           0.0076     0.208    [0.0787, 0.0055]    0.03%

    (true D(0) diag = [0.08, 0.01]).  100 is the useful setting; pushing further
    buys a smaller residual but starts distorting D and sigma, because the wind
    likelihood is then too weak to pin the variance.
    """
    pred = model.heun_step(z, u, dt, dW=None)
    r = z_next - pred

    # Noise-free channels: normalised squared error.  Normalising matters -- the
    # state mixes metres, radians and two momenta, so an unweighted sum would
    # silently optimise whichever channel happens to have the largest units.
    det = ((r[:, :3] / scale[:3]) ** 2).mean()

    # Wind channel: Gaussian NLL with the model's own variance.  sigma must not
    # be free to collapse, so it is bounded below by the integrator noise floor.
    sigma = model.diffusion(z).clamp_min(1e-8)
    var = sigma**2 * dt
    nll = 0.5 * ((r[:, 3] ** 2) / var + torch.log(var)).mean()
    return det_weight * det + nll, det.detach(), nll.detach()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rel(a, b):
    return (torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b).clamp_min(1e-12)).mean().item()


def evaluate(model, p, theta_ref=0.0):
    """Compare the learned functions with the truth, in the same gauge."""
    true = TrueModel(p, theta_ref)
    th = torch.linspace(-np.pi, np.pi, 361, dtype=torch.float64)
    near = torch.linspace(-0.175, 0.175, 51, dtype=torch.float64)   # +/-10 deg

    out = {}
    with torch.no_grad():
        out["M_inv_rel"] = _rel(model.M_inv(th), true.M_inv(th))
        out["M_inv_rel_upright"] = _rel(model.M_inv(near), true.M_inv(near))
        out["D_rel"] = _rel(model.D(th), true.D(th))
        v_hat, v_true = model.potential(th), true.potential(th)
        out["V_rmse"] = torch.sqrt(((v_hat - v_true) ** 2).mean()).item()
        out["V_scale"] = v_true.abs().max().item()
        if model.diffusion_head is not None:
            sp = torch.nn.functional.softplus
            out["sigma_gust"] = sp(model.diffusion_head.raw_gust).item()
            out["sigma_turb"] = sp(model.diffusion_head.raw_turb).item()
    return out


def report(tag, m, p):
    line = (f"{tag:>8}  M^-1 {m['M_inv_rel']:.4f}  (upright {m['M_inv_rel_upright']:.4f})"
            f"  D {m['D_rel']:.4f}  V rmse {m['V_rmse']:.4f} / scale {m['V_scale']:.3f}")
    if "sigma_gust" in m:
        line += (f"\n          sigma_gust {m['sigma_gust']:.5f} (true {p.sigma_gust:.5f})"
                 f"   sigma_turb {m['sigma_turb']:.5f} (true {p.sigma_turb:.5f})")
    print(line)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-traj", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--theta-spread", type=float, default=np.pi,
                    help="initial angles drawn from +/- this; shrink it to "
                         "starve the upright region and reproduce LieSPHGP's "
                         "neural-baseline failure on purpose")
    ap.add_argument("--b-cart", type=float, default=0.08)
    ap.add_argument("--b-base", type=float, default=0.01)
    ap.add_argument("--b-amp", type=float, default=0.02)
    ap.add_argument("--sigma-gust", type=float, default=0.05)
    ap.add_argument("--sigma-turb", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = CartPoleEnv()
    p = replace(Params.from_env(env), b_cart=args.b_cart, b_base=args.b_base,
                b_amp=args.b_amp, sigma_gust=args.sigma_gust, sigma_turb=args.sigma_turb)

    z, u, z_next = make_dataset(p, args.n_traj, args.n_steps, args.dt,
                                args.seed, args.theta_spread, args.obs_noise)
    scale = (z_next - z).std(dim=0).clamp_min(1e-12)
    print(f"dataset: {len(z)} transitions from {args.n_traj} trajectories; "
          f"|theta| <= {z[:,1].abs().max():.2f} rad; "
          f"{(z[:,1].abs() < 0.175).double().mean()*100:.2f}% within 10 deg of upright")
    print(f"true: m_tot {p.m_tot:.4f}  alpha {p.alpha:.4f}  beta {p.beta:.4f}  gamma {p.gamma:.4f}")

    model = PortHamiltonianModel(m_inv_init=1.0, d_init=1e-2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    report("init", evaluate(model, p), p)
    g = torch.Generator().manual_seed(args.seed)
    for k in range(1, args.steps + 1):
        idx = torch.randint(0, len(z), (args.batch,), generator=g)
        loss, det, nll = step_loss(model, z[idx], u[idx], z_next[idx], args.dt, scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if k % max(args.steps // 10, 1) == 0:
            print(f"  step {k:5d}  loss {loss.item():10.4f}  det {det.item():.3e}  nll {nll.item():9.3f}")

    print()
    report("final", evaluate(model, p), p)
    return model


if __name__ == "__main__":
    main()

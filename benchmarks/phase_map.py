"""
Phase-map (phase-portrait) visualization for a trained CT-RL policy on the
pendulum, in support of the Problem-1 "model evaluation" metric.

We roll the deterministic closed-loop system out from a grid of initial
conditions covering the (angle, angular-velocity) plane and plot the resulting
orbits. The upright target (theta = pi) is placed at the origin by plotting the
error angle  phi = wrap(theta - pi)  on the x-axis and  omega = p / I  on the
y-axis. Each orbit is colored by its final phase-distance to the target
(viridis_r: dark = converged). The "in-eps & held" count uses the same
criterion as benchmarks/control_metrics.py: phase distance sqrt(phi^2+omega^2)
< eps sustained over the final `hold` steps, so it is directly comparable to
the eval table's success rate (over grid ICs rather than reset-sampled ICs).

Because ct_sac is model-free there is no learned Hamiltonian to compare; the
phase portrait is the closed-loop flow map Phi_theta^t, which is the level at
which a model-free controller can sit next to the model-learning methods
(PHAST, Lie-SPH) on a shared figure.

Examples
--------
    python -m benchmarks.phase_map --algo ct_sac --save_root save_seeds --seed 0
    python -m benchmarks.phase_map --algo ct_sac \
        --model_path save_seeds/ct_sac/pendulum/top/seed_0/<run>/ct_sac_pendulum_top_200000_steps.pth
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_ct_hyperparams_from_table
from benchmarks.run_ct_rl import make_ct_env
from benchmarks.eval_control_metrics import load_ct_policy


def wrap_to_pi(x: float) -> float:
    return math.remainder(x, 2 * math.pi)


def pick_checkpoint(save_root: str, algo: str, env_id: str, mode: str, seed: int) -> Path:
    """Prefer the 200k-step checkpoint, then final_model, then newest .pth."""
    base = Path(save_root) / algo / env_id / mode / f"seed_{seed}"
    cands = sorted(base.glob("**/*.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No '*.pth' checkpoint under {base}")
    for key in ("200000_steps", "final_model"):
        hit = [c for c in cands if key in c.name]
        if hit:
            return hit[-1]
    return cands[-1]


def set_state(env, theta0: float, omega0: float):
    """Reset the env, then override the physics state to a chosen (theta, omega)."""
    env.reset(seed=0)
    env._theta = wrap_to_pi(theta0)
    env._p = env.I * omega0
    env._sin_theta = np.sin(env._theta)
    env._cos_theta = np.cos(env._theta)
    env._step_index = 0
    env.cur_t = 0.0
    if env.q_only:
        # Pad the position window with the initial measurement (implied v = 0),
        # exactly as the env does on a normal reset.
        env._obs_buf[:] = env._measure(env._theta)
    obs = env._make_obs()
    env._last_obs = np.asarray(obs, dtype=np.float32)
    return obs


def rollout_orbit(env, policy_fn, theta0, omega0, horizon):
    """Closed-loop orbit from (theta0, omega0); returns phi[], omega[] arrays."""
    obs = set_state(env, theta0, omega0)
    phi, om = [], []
    for _ in range(horizon):
        action = np.asarray(policy_fn(obs)).reshape(-1)
        obs, _, term, trunc, _ = env.step(action)
        phi.append(wrap_to_pi(env._theta - env._goal))
        om.append(env._p / env.I)
        if term or trunc:
            break
    return np.asarray(phi), np.asarray(om)


def converged(phi, om, eps, hold):
    """True if the last `hold` samples stay inside the phase-distance eps-ball
    about upright, sqrt(phi^2 + omega^2) < eps — identical to the success
    criterion in benchmarks/control_metrics.py. (A per-axis box test is looser
    in omega and inflates the count.)"""
    if len(phi) < hold:
        return False
    return bool(np.all(np.hypot(phi[-hold:], om[-hold:]) < eps))


def break_wrap(phi, om):
    """Insert NaNs where phi jumps across the +/-pi seam so lines don't streak."""
    phi = phi.copy().astype(float)
    jumps = np.where(np.abs(np.diff(phi)) > math.pi)[0]
    phi_b = np.insert(phi, jumps + 1, np.nan)
    om_b = np.insert(om.astype(float), jumps + 1, np.nan)
    return phi_b, om_b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", default="ct_sac", choices=("ct_sac", "ct_td3", "ct_ddpg"))
    p.add_argument("--env_id", default="pendulum")
    p.add_argument("--mode", default="top")
    p.add_argument("--model_path", default=None)
    p.add_argument("--save_root", default="save_seeds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hyperparams_dir", default="benchmarks/hyperparams")
    p.add_argument("--n_theta", type=int, default=13, help="grid points in angle")
    p.add_argument("--n_omega", type=int, default=9, help="grid points in velocity")
    p.add_argument("--omega_max", type=float, default=8.0)
    p.add_argument("--horizon", type=int, default=None,
                   help="rollout steps per orbit (default: env.max_steps, matching the eval)")
    p.add_argument("--eps", type=float, default=0.1,
                   help="phase-distance convergence radius (matches control_metrics)")
    p.add_argument("--hold", type=int, default=50)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model_path = Path(args.model_path) if args.model_path else pick_checkpoint(
        args.save_root, args.algo, args.env_id, args.mode, args.seed)
    print(f"[phase_map] checkpoint: {model_path}")

    _, env_kwargs, model_kwargs, algo_kwargs, _ = load_ct_hyperparams_from_table(
        algo=args.algo, env_id=args.env_id, mode=args.mode,
        hyperparams_dir=args.hyperparams_dir)
    env = make_ct_env(env_id=args.env_id, seed=args.seed, env_kwargs=env_kwargs)
    policy_fn = load_ct_policy(args.algo, env, model_kwargs, algo_kwargs, model_path)

    # Default to the eval horizon (env.max_steps) so "in-eps & held" judges the
    # same window as eval_control_metrics; cap at max_steps to avoid truncation.
    horizon = args.horizon if args.horizon is not None else (env.max_steps or 800)
    if env.max_steps:
        horizon = min(horizon, env.max_steps)
    # Grid of initial conditions: full angle range, symmetric velocity range.
    thetas = env._goal + np.linspace(-math.pi, math.pi, args.n_theta, endpoint=False)
    omegas = np.linspace(-args.omega_max, args.omega_max, args.n_omega)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap = plt.get_cmap("viridis_r")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=math.pi)

    n_conv = 0
    settle_pts = []  # final points of orbits that came near upright
    for th0 in thetas:
        for om0 in omegas:
            phi, om = rollout_orbit(env, policy_fn, float(th0), float(om0), horizon)
            if len(phi) == 0:
                continue
            final_dist = math.hypot(phi[-1], om[-1])
            n_conv += int(converged(phi, om, args.eps, args.hold))
            if final_dist < 0.5:
                settle_pts.append((phi[-1], om[-1]))
            color = cmap(norm(final_dist))
            phi_b, om_b = break_wrap(phi, om)
            ax.plot(phi_b, om_b, color=color, lw=0.8, alpha=0.75, zorder=1)
            ax.plot(phi[0], om[0], ".", color=color, ms=5, zorder=2)

    # eps-ball, true target, and the empirical attractor the controller reaches.
    ax.add_patch(plt.Circle((0, 0), args.eps, color="k", fill=False,
                            ls="--", lw=1.2, zorder=3))
    ax.plot(0, 0, "*", color="k", ms=16, zorder=5, label="upright target")
    subtitle = f"in-eps & held: {n_conv}/{args.n_theta * args.n_omega}"
    if settle_pts:
        sp = np.asarray(settle_pts)
        # Show every settle point (not just the mean): if the policy has
        # multiple attractors (e.g. offsets on both sides of upright), the mean
        # alone would be misleading — it could even sit at the origin.
        ax.plot(sp[:, 0], sp[:, 1], ".", color="crimson", ms=4, alpha=0.7, zorder=4)
        fp = sp.mean(axis=0)
        spread = float(np.max(np.hypot(sp[:, 0] - fp[0], sp[:, 1] - fp[1])))
        ax.plot(fp[0], fp[1], "X", color="crimson", ms=12, zorder=5,
                label=f"attractor mean ($\\varphi$={fp[0]:+.3f}, $\\omega$={fp[1]:+.3f})")
        subtitle += (f";  settles at dist {math.hypot(*fp):.3f} from upright"
                     f" (spread {spread:.3f})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("final phase-distance to upright")

    ax.set_xlabel(r"angle from upright  $\varphi = \mathrm{wrap}(\theta-\pi)$  [rad]")
    ax.set_ylabel(r"angular velocity  $\omega = p/I$  [rad/s]")
    ax.set_title(f"{args.algo} closed-loop phase map (pendulum, seed {args.seed})\n{subtitle}")
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-args.omega_max, args.omega_max)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = args.out or f"results/phase_map_{args.algo}_seed{args.seed}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[phase_map] in-eps & held: {n_conv}/{args.n_theta * args.n_omega}")
    print(f"[phase_map] saved: {out}")


if __name__ == "__main__":
    main()

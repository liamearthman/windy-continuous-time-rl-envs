"""
Evaluate a trained policy on the pendulum stabilization task using PHAST-comparable
control metrics (success rate, final wrapped-angle error, convergence time, control
effort). Works for both CT-RL algorithms (ct_sac, ct_td3, ...) and SB3 baselines
(sac, td3, ppo, trpo).

The policy is rolled out deterministically on a fresh PendulumEnv built from the same
hyperparam row it was trained on (so dt / horizon / q-only / target all match), and
each rollout is scored with benchmarks/control_metrics.py.

Examples
--------
    # explicit checkpoint
    python -m benchmarks.eval_control_metrics --algo ct_sac \
        --model_path save_qonly/ct_sac/pendulum/top/seed_0/<run>/best_model/best.pth \
        --n_episodes 20 --seed 0

    # auto-find newest checkpoint under a save root
    python -m benchmarks.eval_control_metrics --algo sac \
        --save_root save_qonly --seed 0 --n_episodes 20 --out metrics_sac.json
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)
from benchmarks.run_ct_rl import make_ct_env
from benchmarks.control_metrics import episode_metrics, aggregate

# CT-RL algorithms reload by rebuilding the algorithm then loading a .pth state dict.
CT_ALGOS = ("ct_sac", "ct_td3", "ct_ddpg")
# SB3 algorithms reload from a self-contained .zip.
SB3_ALGOS = ("sac", "td3", "ppo", "trpo")


def find_model(save_root: str, algo: str, env_id: str, mode: str, seed: int, ext: str) -> Path:
    """Final-of-training checkpoint under the run's save dir.

    We deliberately prefer the end-of-training checkpoint (200000_steps, then
    final_model) over best_model.pth. best_model is the peak-*eval-reward*
    snapshot (common/callbacks.py EvalCallback), which (a) is selected on a
    different objective than the reported success/final_error/effort metrics and
    (b) has no analogue in the frozen-after-training Casimir controllers, so
    citing it would give RL an asymmetric cherry-picking advantage. This mirrors
    phase_map.pick_checkpoint so the metrics table and phase maps use the same
    policy. best_model belongs only in the learning-curve view, not the
    head-to-head point comparison.
    """
    base = Path(save_root) / algo / env_id / mode / f"seed_{seed}"
    cands = sorted(base.glob(f"**/*{ext}"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No '*{ext}' checkpoint under {base}")
    for key in ("200000_steps", "final_model"):
        hit = [c for c in cands if key in c.name]
        if hit:
            return hit[-1]
    # Fall back to newest non-best checkpoint, else newest of all.
    nonbest = [c for c in cands if "best" not in c.name.lower()]
    return (nonbest or cands)[-1]


def load_ct_policy(algo, env, model_kwargs, algo_kwargs, model_path) -> Callable:
    """Rebuild a CT-RL algorithm, load weights, return a deterministic policy fn."""
    from models import ActorQCriticModel
    from algorithms.ct_sac import CTSAC
    from algorithms.ct_td3 import CTTD3
    from algorithms.ct_ddpg import CTDDPG

    algo_cls = {"ct_sac": CTSAC, "ct_td3": CTTD3, "ct_ddpg": CTDDPG}[algo]
    # Keep only constructor-accepted kwargs (drops training-only noise_type, etc.).
    accepted = set(inspect.signature(algo_cls.__init__).parameters)
    kwargs = {k: v for k, v in algo_kwargs.items() if k in accepted}

    agent = algo_cls(env=env, model=ActorQCriticModel, model_kwargs=model_kwargs, **kwargs)
    agent.load(str(model_path))
    return lambda obs: agent._policy_act(obs, deterministic=True)


def load_sb3_policy(algo, model_path) -> Callable:
    """Load a SB3 model from .zip and return a deterministic policy fn."""
    from stable_baselines3 import SAC, PPO, TD3
    try:
        from sb3_contrib import TRPO
    except ImportError:
        TRPO = None
    algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO, "trpo": TRPO}[algo]
    if algo_cls is None:
        raise ImportError("sb3_contrib not installed (needed for TRPO).")
    model = algo_cls.load(str(model_path))
    return lambda obs: model.predict(obs, deterministic=True)[0]


def rollout(env, policy_fn: Callable, n_episodes: int, base_seed: int, eps: float):
    """Run n_episodes deterministically and return per-episode metrics."""
    per_ep = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        theta, omega, us = [], [], []
        done = False
        while not done:
            action = np.asarray(policy_fn(obs)).reshape(-1)
            obs, _, term, trunc, _ = env.step(action)
            theta.append(env._theta)           # true wrapped angle
            omega.append(env._p / env.I)        # true angular velocity
            us.append(float(np.clip(action[0], env.min_torque, env.max_torque)))
            done = bool(term or trunc)
        per_ep.append(
            episode_metrics(theta, omega, us, target=env._goal, dt=env.dt, eps=eps)
        )
    return per_ep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", required=True, choices=CT_ALGOS + SB3_ALGOS)
    p.add_argument("--env_id", default="pendulum")
    p.add_argument("--mode", default="top")
    p.add_argument("--model_path", default=None, help="Explicit checkpoint (.pth/.zip).")
    p.add_argument("--save_root", default=None, help="Auto-find newest checkpoint here.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.1, help="Convergence radius.")
    p.add_argument("--hyperparams_dir", default="benchmarks/hyperparams")
    p.add_argument("--out", default=None, help="Write aggregated metrics JSON here.")
    args = p.parse_args()

    is_ct = args.algo in CT_ALGOS
    ext = ".pth" if is_ct else ".zip"

    if args.model_path:
        model_path = Path(args.model_path)
    elif args.save_root:
        model_path = find_model(args.save_root, args.algo, args.env_id, args.mode, args.seed, ext)
        print(f"[auto] using checkpoint: {model_path}")
    else:
        raise SystemExit("Provide --model_path or --save_root.")

    # Build the env from the matching hyperparam row (q-only, dt, horizon, target).
    if is_ct:
        _, env_kwargs, model_kwargs, algo_kwargs, _ = load_ct_hyperparams_from_table(
            algo=args.algo, env_id=args.env_id, mode=args.mode, hyperparams_dir=args.hyperparams_dir)
    else:
        _, env_kwargs, _, _, _ = load_sb3_hyperparams_from_table(
            algo=args.algo, env_id=args.env_id, mode=args.mode, hyperparams_dir=args.hyperparams_dir)

    env = make_ct_env(env_id=args.env_id, seed=args.seed, env_kwargs=env_kwargs)

    if is_ct:
        policy_fn = load_ct_policy(args.algo, env, model_kwargs, algo_kwargs, model_path)
    else:
        policy_fn = load_sb3_policy(args.algo, model_path)

    per_ep = rollout(env, policy_fn, args.n_episodes, args.seed, args.eps)
    agg = aggregate(per_ep)
    agg.update({"algo": args.algo, "env_id": args.env_id, "mode": args.mode,
                "seed": args.seed, "dt": env.dt, "horizon": env.max_steps,
                "target": float(env._goal)})

    print(f"\n=== {args.algo} on {args.env_id} ({args.n_episodes} episodes) ===")
    print(f"  success rate     : {agg['success_rate']:.2f}")
    ct = agg["convergence_time_mean"]
    print(f"  convergence time : {ct:.3f} s" if ct is not None else "  convergence time : (none converged)")
    print(f"  final error      : {agg['final_error_mean']:.4f} +/- {agg['final_error_std']:.4f} rad")
    print(f"  control effort   : {agg['control_effort_mean']:.1f} +/- {agg['control_effort_std']:.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()

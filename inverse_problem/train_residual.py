"""
Train a CT-SAC residual on top of a learned IDA-PBC base law.

    .venv/bin/python -m inverse_problem.train_residual --timesteps 60000

The pipeline, end to end:

  1. identify the plant from trajectory data           (fit_pendulum)
  2. build an IDA-PBC law from the LEARNED parameters  (control.IDAPBC)
  3. train CT-SAC on the residual                      (residual_env)
  4. evaluate base-only against base+residual on the TRUE plant

WHAT THIS IS AND IS NOT.  Not Dyna, not model-based rollouts: the agent trains on
the real environment throughout, and the learned model's only job is to supply a
controller.  That is the claim being tested -- whether identified physics makes a
policy better, as opposed to whether a world model can substitute for data.

THE BASELINES THAT MATTER.  Three, all on the same plant and seeds:

    base-only     max_nudge = 0; pure IDA-PBC from learned physics
    base + RL     the residual policy
    oracle base   IDA-PBC from TRUE parameters, the ceiling for the base law

With wind off, the base law holds upright to under 1e-3 rad; with wind on it
sits at ~0.020 rad.  So essentially all the remaining error is disturbance the
deterministic law cannot anticipate -- sigma(p) = sigma_gust + sigma_turb*|thetadot|
is state-dependent and IDA-PBC is blind to it.  That gap is what the residual
has to close, and 0.020 rad is the number to beat.  A residual that merely
matches the base law has learned nothing; one that beats the ORACLE base law has
learned something the deterministic design cannot express.

A NOTE ON RETURN.  The env's reward penalises total torque, most of which the
base law contributes and the agent cannot influence, so raw return carries a big
agent-independent offset.  Compare the error metrics, not the returns.
"""

from __future__ import annotations

import argparse

import numpy as np

from algorithms.ct_sac import CTSAC
from environment.windypendulumenv import PendulumParams, WindyPendulumEnv
import torch
import torch.nn as nn
from models.actor_q_critic import ActorQCriticModel

from . import control as C
from .fit_pendulum import fit_from_params
from .pendulum import Params
from .residual_env import make_residual_env


# Lifted verbatim from benchmarks/hyperparams/ct_sac.csv, row
# (mode=top, env_id=windypendulum) -- the settings already tuned for this plant.
# Using them rather than inventing new ones keeps the residual run comparable to
# the repo's from-scratch CT-SAC numbers.
MODEL_KWARGS = dict(q_net_arch=[400, 300], pi_net_arch=[400, 300],
                    activation_fn=nn.ReLU,   # the CSV stores "ReLU"; the loader resolves it to the class, log_std_init=-3.0, n_critics=2,
                    deterministic_policy=False, use_actor_target=False)
ALGO_KWARGS = dict(gamma=0.98, buffer_size=300_000, learning_rate=0.00073,
                   batch_size=256, train_freq=1, gradient_steps=1,
                   alpha="auto", tau=0.02, num_expectation_samples=1,
                   target_entropy="auto")


def true_params(dt=0.02) -> Params:
    env = WindyPendulumEnv(params=PendulumParams(), dt=dt, max_steps=200)
    env.reset(seed=0)
    return Params.from_env(env)


def evaluate(env, policy, n_episodes=20, seed=0, target=np.pi, settle_frac=0.5):
    """Roll episodes and score by angle error, not by return.

    `settle_frac` ignores the first half of each episode so the metric measures
    HOLDING upright rather than getting there -- the two are different skills and
    averaging them hides which one changed.
    """
    rms, final, effort, ret = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        errs, us, R = [], [], 0.0
        for _ in range(env.max_steps):
            a = policy(obs, env)
            obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32).reshape(1))
            errs.append(abs(float(C.wrap_to_pi(env._theta - target))))
            us.append(info.get("u_total", 0.0))
            R += r
            if term or trunc:
                break
        k = int(len(errs) * settle_frac)
        rms.append(float(np.sqrt(np.mean(np.square(errs[k:])))))
        final.append(errs[-1])
        effort.append(float(np.sum(np.square(us)) * env.dt))
        ret.append(R)
    return {"rms": float(np.mean(rms)), "rms_sd": float(np.std(rms)),
            "final": float(np.mean(final)), "effort": float(np.mean(effort)),
            "return": float(np.mean(ret))}


def report(tag, m):
    print(f"  {tag:22s} rms {m['rms']:.4f} +/- {m['rms_sd']:.4f} rad   "
          f"final {m['final']:.4f}   effort {m['effort']:7.1f}   return {m['return']:9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--max-nudge", type=float, default=5.0)
    ap.add_argument("--Kp", type=float, default=6.0)
    ap.add_argument("--kd", type=float, default=2.5)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fit-steps", type=int, default=4000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learning-starts", type=int, default=2000,
                    help="the CSV uses 10000; lower it for short smoke runs")
    ap.add_argument("--oracle-base", action="store_true",
                    help="use TRUE parameters in the base law (ablation)")
    args = ap.parse_args()

    P = true_params(args.dt)
    print("identifying the plant ...")
    rep = fit_from_params(P, dt=args.dt, steps=args.fit_steps, seed=args.seed)
    print(rep.table())
    base_params = P if args.oracle_base else rep.params

    common = dict(K_p=args.Kp, k_d=args.kd, dt=args.dt,
                  max_steps=args.max_steps, q_only=False)
    train_env = make_residual_env(base_params, max_nudge=args.max_nudge,
                                  seed=args.seed, **common)
    eval_env = make_residual_env(base_params, max_nudge=args.max_nudge,
                                 seed=args.seed + 500, **common)
    base_env = make_residual_env(base_params, max_nudge=0.0,
                                 seed=args.seed + 500, **common)
    oracle_env = make_residual_env(P, max_nudge=0.0,
                                   seed=args.seed + 500, **common)

    zero = lambda obs, env: np.zeros(1, dtype=np.float32)
    print(f"\nbaselines ({args.episodes} episodes, metrics over the last "
          f"{int((1-0.5)*100)}% of each)")
    report("oracle base (no RL)", evaluate(oracle_env, zero, args.episodes, args.seed))
    report("learned base (no RL)", evaluate(base_env, zero, args.episodes, args.seed))

    print(f"\ntraining CT-SAC residual for {args.timesteps} steps ...")
    algo = CTSAC(env=train_env, model=ActorQCriticModel, seed=args.seed,
                 model_kwargs=MODEL_KWARGS,
                 learning_starts=args.learning_starts, **ALGO_KWARGS)
    algo.learn(total_timesteps=args.timesteps, log_interval=max(args.timesteps // 5, 1))

    def trained(obs, env):
        # CTSAC has no .predict; the policy lives on the model, and act() returns
        # (actions, log_prob) already rescaled to the env's action box.
        with torch.no_grad():
            a, _ = algo.model.act(
                torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1),
                deterministic=True)
        return a.cpu().numpy().reshape(-1)

    print()
    report("learned base + RL", evaluate(eval_env, trained, args.episodes, args.seed))
    return algo


if __name__ == "__main__":
    main()

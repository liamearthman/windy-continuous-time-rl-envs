#!/usr/bin/env python3
"""Inspect a dm_control env: spec dump, flattened obs index->name layout,
and the true MuJoCo state (qpos/qvel) for comparison.

Usage:
    python inspect_env.py                      # cheetah / run
    python inspect_env.py walker walk
    python inspect_env.py cheetah run --seed 0
"""
from __future__ import annotations

import argparse
import numpy as np
from dm_control import suite


def build_layout(obs_spec):
    """Reconstruct flattened obs layout using the SAME sorted-key order
    that the wrapper's _flatten_obs uses."""
    layout, i = [], 0
    for k in sorted(obs_spec.keys()):
        n = int(np.prod(obs_spec[k].shape))  # np.prod(()) == 1, matches .ravel()
        layout.append((k, i, i + n))
        i += n
    return layout, i


def main():
    p = argparse.ArgumentParser()

    p.add_argument("domain", nargs="?", default="quadruped")
    p.add_argument("task", nargs="?", default="run")
    p.add_argument("--seed", type=int, default=None)


    args = p.parse_args()

    task_kwargs = {"random": args.seed} if args.seed is not None else {}
    env = suite.load(args.domain, args.task, task_kwargs=task_kwargs)

    print(f"=== {args.domain} / {args.task} ===\n")

    # 1. Specs
    aspec = env.action_spec()
    print(f"ACTION  shape={aspec.shape} dtype={aspec.dtype} "
          f"range=[{np.min(aspec.minimum)}, {np.max(aspec.maximum)}]\n")

    print("OBSERVATION SPEC:")
    for k in sorted(env.observation_spec().keys()):
        v = env.observation_spec()[k]
        print(f"  {k:<16} shape={v.shape}")
    print()

    # 2. Flattened layout (index -> name)
    layout, total = build_layout(env.observation_spec())
    print(f"FLATTENED OBS LAYOUT (total dim = {total}):")
    for name, lo, hi in layout:
        print(f"  obs[{lo}:{hi}] -> {name}")
    print()


    # 2.5 
    physics = env.physics
    nu, nq, nv = physics.model.nu, physics.model.nq, physics.model.nv
    print(f"MODEL DIMENSIONS:")
    print(f"  nq (qpos): {nq}")
    print(f"  nv (qvel): {nv}")
    print(f"  nu (action): {nu}")
    print(f"  nq - nu = {nq - nu}   (root joint dim in qpos)")
    print(f"  nv - nu = {nv - nu}   (underactuation gap)")
    print()

    print(f"ACTUATORS (n={nu}):")
    for i in range(nu):
        try:
            name = physics.model.id2name(i, 'actuator')
        except AttributeError:
            name = physics.model.actuator(i).name  # newer dm_control API
        print(f"  action[{i}] -> '{name}'")
    print()

    # 3. A reset + rollout over several steps, showing obs AND true state
    N_STEPS = 3

    def flatten(obs):
        """Match the wrapper's _flatten_obs: sorted-key concatenation."""
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(obs)]
        )

    def show_state(tag, ts):
        print(f"--- {tag} ---")
        print(f"  flattened obs (dim {flatten(ts.observation).shape[0]}):")
        print(f"    {flatten(ts.observation)}")
        print(f"  qpos (dim {env.physics.data.qpos.shape[0]}):")
        print(f"    {env.physics.named.data.qpos}")
        print(f"  qvel (dim {env.physics.data.qvel.shape[0]}):")
        print(f"    {env.physics.named.data.qvel}")
        r = "None" if ts.reward is None else f"{ts.reward:.6f}"
        print(f"  reward={r}  discount={ts.discount}  step_type={ts.step_type}\n")

    ts = env.reset()
    show_state("RESET", ts)

    action = np.zeros(aspec.shape)  # hold zero so drift is pure dynamics
    for i in range(N_STEPS):
        ts = env.step(action)
        show_state(f"STEP {i + 1}", ts)

    print(f"state dim (qpos+qvel) = "
          f"{env.physics.data.qpos.shape[0] + env.physics.data.qvel.shape[0]}")
    print(f"flattened obs dim     = {total}")


if __name__ == "__main__":
    main()
"""
Residual-RL environment: a learned IDA-PBC base law with an RL nudge on top.

    u = clip( u_IDA-PBC(theta, p ; learned params) + nudge )

The agent chooses only `nudge`.  The base law does the stabilising; the policy
learns the correction -- for wind, which the deterministic law cannot anticipate,
and for whatever the identified model got wrong.

WHY A WRAPPER AND NOT A NEW ALGORITHM.  CTSAC is untouched.  Subclassing the env
and overriding `_step_physics` is the whole change: observations, reward,
episode handling, the integrator and the time-aware `step_dt` plumbing are all
inherited, so a residual run and a from-scratch run differ in exactly one place
and are therefore comparable.

ACTION SPACE.  Box(-max_nudge, +max_nudge).  models/actor.py tanh-squashes and
rescales from action_space.low/high, so declaring the bound here is all that is
needed -- no scaling code, and the agent cannot exceed the budget by construction.

THE BOUND IS THE DESIGN.  An unbounded residual can cancel the base law outright,
at which point this is plain model-free RL wearing a costume.  Bounded, the
policy can only correct, and the comparison "did the learned physics help?" stays
meaningful.  max_nudge = 0 reduces the env to the pure base controller, which is
how the no-RL baseline is run -- same code path, same seeds.

REWARD.  Inherited unchanged, and computed on the TOTAL torque.  That means the
effort penalty covers the base law's torque too, most of which the agent cannot
influence.  That is the physically honest choice (the plant pays for all of it)
but it does put a large agent-independent offset in the return, so compare
policies by the error metrics, not by raw return.

SENSING.  The base law is state feedback and reads the env's true (theta, p).
Under q_only=True the AGENT sees only a window of angle measurements while the
base law still sees the state -- privileged information, and not a fair
comparison against a q-only baseline.  Fine for the full-state setting the
identification was done in; flagged because it is easy to miss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from environment.windypendulumenv import PendulumParams, WindyPendulumEnv

from .control import IDAPBC
from .pendulum import Params


class ResidualIDAPBCPendulum(WindyPendulumEnv):
    """WindyPendulumEnv where the action is a residual on an IDA-PBC base law."""

    def __init__(self, *, base_params: Params, K_p: float = 6.0, k_d: float = 2.5,
                 max_nudge: float = 5.0, cancel_damping: bool = True,
                 params: Optional[PendulumParams] = None, **env_kwargs):
        super().__init__(params=params or PendulumParams(), **env_kwargs)
        self.base = IDAPBC(base_params, K_p=K_p, k_d=k_d,
                           max_torque=self.max_torque, cancel_damping=cancel_damping)
        self.max_nudge = float(max_nudge)
        # Overrides the parent's +/-max_torque box: the agent's action is the
        # NUDGE, not the torque.
        self.action_space = spaces.Box(
            low=-self.max_nudge, high=self.max_nudge, shape=(1,), dtype=np.float32)
        self._last = {"u_base": 0.0, "nudge": 0.0, "u_total": 0.0}

    def base_action(self) -> float:
        """The base law's torque at the current state, unclipped."""
        return self.base.act([self._theta, self._p], clip=False)

    def _step_physics(self, action, dt):
        nudge = float(np.clip(np.asarray(action).reshape(-1)[0],
                              -self.max_nudge, self.max_nudge))
        u_base = self.base_action()
        # The parent clips the total to +/-max_torque; doing it once there keeps
        # one clip in the pipeline rather than two that can disagree.
        out = super()._step_physics(np.array([u_base + nudge]), dt)
        self._last = {"u_base": u_base, "nudge": nudge,
                      "u_total": float(np.clip(u_base + nudge,
                                               self.min_torque, self.max_torque))}
        out[4].update(self._last)
        return out


def make_residual_env(base_params: Params, *, max_nudge=5.0, K_p=6.0, k_d=2.5,
                      dt=0.02, max_steps=500, q_only=False, seed=None,
                      **kw) -> ResidualIDAPBCPendulum:
    env = ResidualIDAPBCPendulum(
        base_params=base_params, K_p=K_p, k_d=k_d, max_nudge=max_nudge,
        dt=dt, max_steps=max_steps, q_only=q_only, **kw)
    if seed is not None:
        env.reset(seed=seed)
    return env

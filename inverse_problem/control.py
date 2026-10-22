"""
IDA-PBC for the windy pendulum, driven by learned or true physics.

THE LAW.  The pendulum is fully actuated, so interconnection-and-damping
assignment reduces to energy shaping plus damping injection -- no matching PDE.
Plant and target:

    plant     pdot = -gamma sin(theta) + u - D(theta) thetadot
    target    pdot = -dV_d/dtheta - k_d thetadot,    V_d = -K_p cos(theta - pi)

Matching the two and using sin(theta - pi) = -sin(theta):

    u = (gamma + K_p) sin(theta) + (D(theta) - k_d) thetadot            (*)

The closed-loop energy is H_d = p^2/(2I) + K_p cos(theta), whose minimum is at
theta = pi -- upright, the env's `_goal`.  Along closed-loop trajectories
Hdot_d = -k_d thetadot^2 <= 0, so H_d is a Lyapunov function and LaSalle gives
asymptotic convergence to upright in the deterministic case.  Hanging, which was
the plant's stable equilibrium, becomes the MAXIMUM of V_d and is repelling.

WHICH TERMS CARRY LEARNED PHYSICS.  All of them:

    gamma    sets how much torque is needed to beat gravity.  Too small and the
             pendulum stalls short of upright; too large and it overshoots.
    D(theta) is CANCELLED and replaced by k_d.  An error here is injected
             directly as spurious damping -- this is the term that makes the
             cart-pole's 12% error on D a control problem rather than a
             cosmetic one.
    I        enters through thetadot = p/I, scaling the whole damping channel.

So (*) is simultaneously the controller and the sharpest test of the fit: an
open-loop prediction error that looks small can still flip a sign here, which is
exactly the failure LieSPHGP reports for its neural baseline.

TORQUE BUDGET.  Holding theta = pi/2 against gravity needs |u| = gamma + K_p.
With gamma = 9.81 and the env's +/-20 limit, K_p must stay below about 10 or the
law saturates on the way up and the Lyapunov argument no longer applies.
`feasible_Kp` makes that explicit rather than leaving it to be discovered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import pendulum as pend
from .pendulum import MAX_TORQUE, UPRIGHT, Params


def wrap_to_pi(theta):
    """Angle error folded into [-pi, pi).  Note the half-open end: exactly pi
    maps to -pi, which is the same point on the circle and harmless because
    callers take |.|.  sin() in the law is periodic, so wrapping is only needed
    for MEASURING error -- but measuring it wrong makes a successful run look
    like a failure near the wrap point."""
    return np.remainder(theta + np.pi, 2 * np.pi) - np.pi


def feasible_Kp(p: Params, max_torque: float = MAX_TORQUE, margin: float = 0.75) -> float:
    """Largest K_p that keeps the worst-case shaping torque inside the limit."""
    return max(margin * max_torque - p.gamma, 0.0)


@dataclass
class IDAPBC:
    """The control law (*).  `params` may be true or learned -- that is the point."""

    params: Params
    K_p: float = 6.0
    k_d: float = 1.2
    target: float = UPRIGHT
    max_torque: float = MAX_TORQUE
    cancel_damping: bool = True

    def thetadot(self, z) -> float:
        return float(z[1]) / self.params.inertia

    def energy(self, z) -> float:
        """H_d, the closed-loop Lyapunov function."""
        th, thd = float(z[0]), self.thetadot(z)
        return 0.5 * self.params.inertia * thd**2 - self.K_p * math.cos(th - self.target)

    def act(self, z, clip: bool = True) -> float:
        th, thd = float(z[0]), self.thetadot(z)
        shaping = (self.params.gamma + self.K_p) * math.sin(th - self.target + math.pi)
        damp = -self.k_d * thd
        if self.cancel_damping:
            damp += pend.damping_raw(th, self.params) * thd
        u = shaping + damp
        return float(np.clip(u, -self.max_torque, self.max_torque)) if clip else float(u)


# ---------------------------------------------------------------------------
# Closed-loop evaluation on the TRUE plant
# ---------------------------------------------------------------------------

@dataclass
class ClosedLoopResult:
    t: np.ndarray
    z: np.ndarray
    u: np.ndarray
    err: np.ndarray            # |wrap(theta - target)|
    settled_at: Optional[float]
    final_err: float
    rms_err_last: float
    effort: float
    saturation: float

    def summary(self) -> str:
        s = "never" if self.settled_at is None else f"{self.settled_at:.2f}s"
        return (f"settle {s}  final |err| {self.final_err:.4f} rad  "
                f"rms(last 2s) {self.rms_err_last:.4f}  "
                f"effort {self.effort:.1f}  sat {self.saturation*100:.1f}%")


def run_closed_loop(policy: Callable[[np.ndarray], float], true_params: Params,
                    z0, n_steps=1500, dt=0.02, seed=0, target=UPRIGHT,
                    tol=0.1, hold=0.5) -> ClosedLoopResult:
    """Simulate the LEARNED controller against the TRUE plant.

    The split matters: the policy may hold a learned model, but the dynamics it
    acts on are always the real ones, wind included.  A controller evaluated
    inside its own model proves nothing.

    `settled_at` is the first time |error| drops below `tol` and STAYS there for
    `hold` seconds -- touch-once is not stabilisation, the same definition the
    repo's control_metrics already uses for the PHAST comparison.
    """
    rng = np.random.default_rng(seed)
    z = np.empty((n_steps + 1, pend.N_STATE))
    z[0] = np.asarray(z0, dtype=float)
    u = np.empty(n_steps)
    for k in range(n_steps):
        u[k] = np.clip(policy(z[k]), -MAX_TORQUE, MAX_TORQUE)
        dW = math.sqrt(dt) * rng.standard_normal()
        z[k + 1] = pend.heun_step(z[k], u[k], dW, dt, true_params)

    err = np.abs(wrap_to_pi(z[:, 0] - target))
    hold_n = int(round(hold / dt))
    settled = None
    inside = err < tol
    for i in range(len(inside) - hold_n):
        if inside[i:i + hold_n + 1].all():
            settled = i * dt
            break
    last = max(len(err) - int(round(2.0 / dt)), 0)
    return ClosedLoopResult(
        t=np.arange(len(z)) * dt, z=z, u=u, err=err, settled_at=settled,
        final_err=float(err[-1]),
        rms_err_last=float(np.sqrt((err[last:] ** 2).mean())),
        effort=float((u**2).sum() * dt),
        saturation=float((np.abs(u) >= MAX_TORQUE - 1e-9).mean()),
    )


def residual_policy(base: IDAPBC, nudge: Callable[[np.ndarray], float],
                    max_nudge: float = 5.0) -> Callable[[np.ndarray], float]:
    """u = u_IDA-PBC(z) + clip(nudge(z)).

    The RL layer.  Bounding the nudge is what keeps the arrangement meaningful:
    an unbounded residual can erase the base law entirely, at which point this
    is plain model-free RL with extra steps.  A bounded one can only correct --
    for wind, and for the model error the base law cannot see.
    """
    def pi(z):
        return base.act(z, clip=False) + float(np.clip(nudge(z), -max_nudge, max_nudge))
    return pi

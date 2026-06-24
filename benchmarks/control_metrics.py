"""
Control metrics for the pendulum stabilization task, defined to match the PHAST
Energy-Casimir control study (Table 7 / Sec. 5):

  - success        : trajectory reaches the eps-ball around the target AND
                     stays inside it for the final `stability_window` steps,
                     sqrt(wrap(theta - q*)^2 + omega^2) < eps   (eps = 0.1)
  - final_error    : sqrt(wrap(theta_T - q*)^2 + omega_T^2)     (phase distance)
  - convergence_time: first time (seconds) the eps-ball is reached, else None
  - control_effort : sum_t u_t^2 * dt

These operate on a plain (theta, omega, u) rollout, so they are identical
whether the controller is an RL policy or the Casimir controller.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def wrap_to_pi(a):
    """Wrap angle(s) to [-pi, pi]."""
    return np.remainder(np.asarray(a) + np.pi, 2 * np.pi) - np.pi


def episode_metrics(
    theta: np.ndarray,
    omega: np.ndarray,
    u: np.ndarray,
    target: float = np.pi,
    dt: float = 0.01,
    eps: float = 0.1,
    stability_window: int = 50,
) -> Dict[str, Optional[float]]:
    """Compute control metrics for a single rollout.

    Args:
        theta, omega, u: per-step true angle, angular velocity, applied torque.
        target: regulation target q* (default pi, the upright).
        dt: timestep (for convergence time in seconds).
        eps: convergence radius in phase space.
        stability_window: a trial succeeds only if the last `stability_window`
            steps all stay within the eps-ball (matches the PHAST benchmark).
    """
    theta = np.asarray(theta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    u = np.asarray(u, dtype=float)

    ang_err = wrap_to_pi(theta - target)
    # Phase-space distance, matching PHAST: sqrt(wrap(q - q*)^2 + p^2).
    # (omega == p here since the pendulum has unit inertia I = m L^2 = 1.)
    dist = np.sqrt(ang_err ** 2 + omega ** 2)

    below = np.flatnonzero(dist < eps)
    conv_step = int(below[0]) if below.size else None

    # PHAST: success requires staying in the eps-ball for the final window.
    window = min(stability_window, dist.size)
    stayed = bool(np.all(dist[-window:] < eps)) if window > 0 else False
    success = conv_step is not None and stayed

    return {
        "final_error": float(dist[-1]),
        "convergence_time": (float(conv_step * dt) if conv_step is not None else None),
        "success": bool(success),
        "control_effort": float(np.sum((u ** 2) * dt)),
    }


def aggregate(per_episode: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """Mean/std over episodes, plus success rate (PHAST-style reporting)."""
    out: Dict[str, Optional[float]] = {"n_episodes": len(per_episode)}
    for key in ("final_error", "control_effort"):
        vals = np.array([m[key] for m in per_episode], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())

    out["success_rate"] = float(np.mean([m["success"] for m in per_episode]))

    conv = [m["convergence_time"] for m in per_episode if m["convergence_time"] is not None]
    out["convergence_time_mean"] = float(np.mean(conv)) if conv else None
    out["convergence_time_std"] = float(np.std(conv)) if conv else None
    return out

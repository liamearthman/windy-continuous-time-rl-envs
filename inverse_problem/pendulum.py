"""
Forward model and synthetic data for windy-pendulum system identification.

The cart-pole analogue of this file is dynamics.py + data.py; this is the same
job on a plant that is easier in three specific ways, all of which matter for
the controller we are building toward:

  FULLY ACTUATED.  u is a torque on p directly, so IDA-PBC here is energy
  shaping plus damping injection -- no matching PDE, no kinetic-energy shaping.
  PHAST/ida_pbc_cartpole.py's docstring explains the contrast: on the cart-pole
  the instability lives in the UNACTUATED coordinate, which is what forces full
  IDA-PBC there.  Not so here.

  M IS A CONSTANT.  I = m L^2 with no theta dependence, so "learning M^-1" is
  learning one positive number.  On the cart-pole M(theta) and V(theta) both
  feed dH/dtheta and separate only through the p-dependence; that whole
  difficulty disappears.

  DAMPING IS BIG.  b_base = 0.3, b_amplitude = 0.5 against the 0.01-0.08 added
  to the cart-pole.  The damping torque is then ~10-20% of the gravity torque
  rather than ~3.6%, so D -- the one quantity the cart-pole fit recovered badly
  (12% error) -- has far better signal-to-noise.

CONVENTION, AND IT IS THE OPPOSITE OF THE CART-POLE.  theta = 0 is HANGING and
theta = pi is UPRIGHT (WindyPendulumEnv._goal = np.pi).  V is therefore minimal
at 0 and maximal at the target.  Getting this backwards silently inverts the
controller, so every angle here is in the env's convention.

COORDINATES.  The dynamics depend on (m, L, g) only through

    I = m L^2        (inertia)
    gamma = m g L    (gravity torque scale)

With g known these invert to L = I g / gamma and m = gamma^2 / (g^2 I); with g
unknown only I and gamma are identifiable.  Same reasoning as the cart-pole's
combo coordinates: fit what the data sees, convert at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

PARAM_NAMES = ("inertia", "gamma", "b_base", "b_amp", "sigma_gust", "sigma_turb")
MECHANICAL = (0, 1)
DAMPING = (2, 3)
WIND = (4, 5)
N_STATE = 2
N_PARAM = 6

# Wind acts on p only -- rank-one diffusion in a 2-D state, same structure as
# the cart-pole, so again three... here ONE channel is noise-free (theta) and
# one carries all the noise (p).
G_INPUT = np.array([0.0, 1.0])
G_NOISE = np.array([0.0, 1.0])
J_PH = np.array([[0.0, 1.0], [-1.0, 0.0]])

UPRIGHT = np.pi
HANGING = 0.0
MAX_TORQUE = 20.0


@dataclass(frozen=True)
class Params:
    inertia: float
    gamma: float
    b_base: float = 0.0
    b_amp: float = 0.0
    sigma_gust: float = 0.0
    sigma_turb: float = 0.0

    @classmethod
    def from_physical(cls, m, L, g, b_base=0.0, b_amp=0.0,
                      sigma_gust=0.0, sigma_turb=0.0) -> "Params":
        return cls(inertia=m * L**2, gamma=m * g * L, b_base=b_base, b_amp=b_amp,
                   sigma_gust=sigma_gust, sigma_turb=sigma_turb)

    @classmethod
    def from_env(cls, env) -> "Params":
        p = env.params
        return cls.from_physical(p.m, p.L, p.g, p.b_base, p.b_amplitude,
                                 env.sigma_gust, env.sigma_turb)

    def to_physical(self, g: float = 9.81) -> dict:
        L = self.inertia * g / self.gamma
        return {"m": self.gamma**2 / (g**2 * self.inertia), "L": L, "g": g,
                "b_base": self.b_base, "b_amp": self.b_amp}

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    @classmethod
    def from_vector(cls, v) -> "Params":
        return cls(*(float(x) for x in np.asarray(v, dtype=float)))

    def perturbed(self, index: int, delta: float) -> "Params":
        n = PARAM_NAMES[index]
        return replace(self, **{n: getattr(self, n) + delta})


# ---------------------------------------------------------------------------
# Physics.  `xp` lets one source serve numpy and torch, as in dynamics.py.
# ---------------------------------------------------------------------------

def damping_raw(theta, p: Params, xp=np):
    """D(theta) = b_base + b_amp*|sin theta|.

    NOT smoothed, unlike the cart-pole version.  WindyPendulumEnv uses an exact
    np.abs and this module must be the same map; the kink sits at theta = 0 and
    pi, i.e. hanging and upright.  It is a genuine non-differentiability at the
    control target -- worth remembering when the learned D feeds a controller --
    but introducing a smoothing the plant does not have would be worse.
    """
    return p.b_base + p.b_amp * xp.abs(xp.sin(theta))


def drift_raw(z, u, p: Params, xp=np):
    """(thetadot, pdot).  Matches WindyPendulumEnv.force exactly."""
    theta, mom = z[0], z[1]
    thetadot = mom / p.inertia
    pdot = -p.gamma * xp.sin(theta) + u - damping_raw(theta, p, xp) * thetadot
    return thetadot, pdot


def diffusion_raw(z, p: Params, xp=np):
    """sigma = sigma_gust + sigma_turb*|thetadot|.  Depends on p only."""
    return p.sigma_gust + p.sigma_turb * xp.abs(z[1] / p.inertia)


def drift(z, u, p: Params) -> np.ndarray:
    return np.array(drift_raw(z, u, p), dtype=float)


def diffusion(z, p: Params) -> float:
    return float(diffusion_raw(z, p))


def heun_step(z, u, dW, dt, p: Params) -> np.ndarray:
    """One stochastic-Heun step, noise supplied rather than drawn.

    Mirrors WindyPendulumEnv.integrator, including the detail that the THETA
    predictor uses the current p (a plain Euler half) while the corrector
    averages current and predicted thetadot.  That asymmetry is in the env; do
    not "fix" it here or the two maps stop agreeing.
    """
    z = np.asarray(z, dtype=float)
    theta, mom = z[0], z[1]
    f0 = drift(z, u, p)
    s0 = diffusion(z, p)
    theta_t = theta + (mom / p.inertia) * dt
    p_t = mom + f0[1] * dt + s0 * dW
    z_t = np.array([theta_t, p_t])
    f1 = drift(z_t, u, p)
    s1 = diffusion(z_t, p)
    return np.array([
        theta + 0.5 * (mom / p.inertia + p_t / p.inertia) * dt,
        mom + 0.5 * (f0[1] + f1[1]) * dt + 0.5 * (s0 + s1) * dW,
    ])


# ---------------------------------------------------------------------------
# Port-Hamiltonian layer
# ---------------------------------------------------------------------------

def hamiltonian_raw(z, p: Params, xp=np):
    """H = p^2/(2I) + V(theta), V = -gamma*cos(theta).

    V is minimal hanging (theta = 0) and maximal upright (theta = pi), which is
    the env's convention.  The additive constant is a gauge; -gamma*cos fixes it
    so that V(0) = -gamma.
    """
    return z[1] ** 2 / (2.0 * p.inertia) - p.gamma * xp.cos(z[0])


def grad_hamiltonian_raw(z, p: Params, xp=np):
    return (p.gamma * xp.sin(z[0]), z[1] / p.inertia)


def hamiltonian(z, p: Params) -> float:
    return float(hamiltonian_raw(z, p))


def grad_hamiltonian(z, p: Params) -> np.ndarray:
    return np.array(grad_hamiltonian_raw(z, p), dtype=float)


def dissipation_matrix(z, p: Params) -> np.ndarray:
    """R = diag(0, D(theta)) -- momentum block only, as always."""
    R = np.zeros((N_STATE, N_STATE))
    R[1, 1] = damping_raw(z[0], p)
    return R


def drift_from_ph(z, u, p: Params) -> np.ndarray:
    """f = (J - R) grad H + G u.  Identical to `drift`."""
    return (J_PH - dissipation_matrix(z, p)) @ grad_hamiltonian(z, p) + G_INPUT * u


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    t: np.ndarray
    z: np.ndarray             # (n+1, 2) true states
    u: np.ndarray             # (n,) applied torque, clipped
    dW: np.ndarray            # (n,) noise increments used
    obs: np.ndarray
    dt: float
    params: Params
    obs_noise_std: np.ndarray
    seed: Optional[int]

    @property
    def n_steps(self) -> int:
        return len(self.u)


def rollout(params: Params, z0, u, dt, rng=None, dW=None,
            max_torque: Optional[float] = MAX_TORQUE):
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    if max_torque is not None:
        u = np.clip(u, -max_torque, max_torque)
    if dW is None:
        if rng is None:
            raise ValueError("rollout needs either dW or rng")
        dW = np.sqrt(dt) * rng.standard_normal(n)
    else:
        dW = np.asarray(dW, dtype=float)
        if len(dW) != n:
            raise ValueError(f"dW has length {len(dW)}, expected {n}")
    z = np.empty((n + 1, N_STATE))
    z[0] = np.asarray(z0, dtype=float)
    for k in range(n):
        z[k + 1] = heun_step(z[k], u[k], dW[k], dt, params)
    return z, u, dW


def input_random_steps(n_steps, dt, rng, amplitude=8.0, hold_seconds=0.2):
    hold = max(int(round(hold_seconds / dt)), 1)
    blocks = rng.uniform(-amplitude, amplitude, size=int(np.ceil(n_steps / hold)))
    return np.repeat(blocks, hold)[:n_steps]


def generate(params: Params, n_steps=500, dt=0.02, z0=None, amplitude=8.0,
             obs_noise_std=0.0, seed=0) -> Trajectory:
    rng = np.random.default_rng(seed)
    u = input_random_steps(n_steps, dt, rng, amplitude)
    if z0 is None:
        z0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-3.0, 3.0)])
    z, u, dW = rollout(params, z0, u, dt, rng=rng)
    std = np.broadcast_to(np.atleast_1d(np.asarray(obs_noise_std, float)),
                          (N_STATE,)).astype(float).copy()
    obs = z.copy() if np.all(std == 0) else z + rng.standard_normal(z.shape) * std
    return Trajectory(t=np.arange(len(z)) * dt, z=z, u=u, dW=dW, obs=obs, dt=dt,
                      params=params, obs_noise_std=std, seed=seed)


def energy_series(traj: Trajectory, params: Optional[Params] = None) -> np.ndarray:
    p = params or traj.params
    return np.array([hamiltonian(zk, p) for zk in traj.z])

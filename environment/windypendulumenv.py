from __future__ import annotations
from .base import ContinuousEnv
import random 

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import math

def wrap_to_pi(theta):
    return math.remainder(theta, 2 * math.pi)


class PendulumParams:
    """physical parameters of the pendulum"""
    m: float = 1.0
    L: float = 1.0
    g: float = 9.81
    b_base: float = 0.3
    b_amplitude: float = 0.5
    sigma_gust: float = 0.1
    sigma_turb: float = 0.05



class WindyPendulumEnv(ContinuousEnv):
    metadata = {"render_modes": []}

    def __init__(self, *, dt: float = 0.02, max_torque: float = 20.0, min_torque: float = -20.0,
                params: PendulumParams, max_steps: int = 200, q_only: bool = True,
                sigma_gust: float = 0.1, sigma_turb: float = 0.5,
                obs_window: int = 4, reward_version: str = "v0",
                hold_eps: float = 0.1, hold_bonus: float = 1.0,
                init_mode: str = "near_upright"):
        super().__init__(dt=dt, max_steps=int(max_steps))
        # Initial-condition distribution.
        #   "near_upright" -> theta ~ pi + N(0, 0.3) (PHAST-comparable stabilization).
        #   "uniform"      -> theta ~ U(-pi, pi): swing-up from ANY angle (incl.
        #                     hanging), a much harder task than local stabilization.
        self.init_mode = str(init_mode)
        # Reward shaping. "v0" = original smooth quadratic. "v1" = v0 plus a
        # per-step holding bonus of `hold_bonus` whenever the phase-distance to
        # upright sqrt(phi^2 + omega^2) is within `hold_eps` (matches the eval's
        # sustained-hold criterion, so training is aligned with what we score).
        self.reward_version = str(reward_version)
        self.hold_eps = float(hold_eps)
        self.hold_bonus = float(hold_bonus)
        self.dt = float(dt)
        self.max_torque = float(max_torque)
        self.min_torque = float(min_torque)
        self.params = params
        self.b_base = self.params.b_base
        self.b_amplitude = self.params.b_amplitude
        self.sigma_gust = float(sigma_gust)
        self.sigma_turb = float(sigma_turb)
        # Wind strength: CSV/kwarg override takes precedence over the params default,
        # so a sigma sweep can vary it per run without editing the params class.

        # Observation mode.
        #   q_only=False -> full state [cos(theta), sin(theta), p].
        #   q_only=True  -> PHAST-style position-only sensing: a window of the
        #                   last `obs_window` (optionally noisy) position
        #                   measurements [cos, sin], with momentum hidden. The
        #                   agent must infer velocity from the position history.
        self.q_only = bool(q_only)
        self.obs_window = int(obs_window)
        self._obs_buf = np.zeros((self.obs_window, 2), dtype=np.float32)

        self.I = self.params.m * self.params.L**2

        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        if self.q_only:
            # cos/sin are always in [-1, 1]; window holds `obs_window` frames.
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2 * self.obs_window,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # episode state
        self._theta = random.uniform(-np.pi, np.pi)
        # Stabilization target: the upright (theta = pi), matching the fixed
        # target q* = pi in the PHAST Casimir control task.
        self._goal = np.pi
        self._p = 0.0
        self._step_count = 0
        self._sin_theta = np.sin(self._theta)
        self._cos_theta = np.cos(self._theta)


    def compute_damping(self, theta):
        return self.b_base + self.b_amplitude * np.abs(np.sin(theta))

    def _measure(self, theta):
        """A (cos, sin) position measurement of `theta` (q-only sensing)."""
        return np.float32(np.cos(theta)), np.float32(np.sin(theta))

    def _push_measurement(self, theta):
        """Append a new position measurement to the sliding window (oldest first)."""
        self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
        self._obs_buf[-1] = self._measure(theta)

    def _make_obs(self):
        """Build the observation for the current mode."""
        if self.q_only:
            return self._obs_buf.reshape(-1).astype(np.float32)
        return np.array([self._cos_theta, self._sin_theta, self._p], dtype=np.float32)


    def _reset_physics(self, *, seed, options):
        # Use the per-env RNG seeded by ContinuousEnv.reset(seed=...) so that
        # different --seed values give different, reproducible initial states
        # (instead of drawing from the global `random` module).
        #
        # IC per init_mode. Both draw from the seeded per-env RNG (self._np_random)
        # so the ICs are reproducible under the env seed.
        #   near_upright: theta ~ pi + N(0, 0.3) -- stabilization about upright,
        #                 matching the PHAST Casimir initialization.
        #   uniform:      theta ~ U(-pi, pi) -- swing-up from any angle, incl.
        #                 hanging; a much harder task than local stabilization.
        if self.init_mode == "uniform":
            self._theta = float(self._np_random.uniform(-np.pi, np.pi))
        else:
            self._theta = float(np.pi + self._np_random.normal(0.0, 0.3))
        omega0 = float(self._np_random.normal(0.0, 0.5))
        self._p = self.I * omega0

        self._sin_theta = np.sin(self._theta)
        self._cos_theta = np.cos(self._theta)
        self._step_count = 0

        if self.q_only:
            # Pad the window with the initial measurement (so the implied
            # velocity starts at zero), matching PHAST's q_hist = q0.repeat(W).
            self._obs_buf[:] = self._measure(self._theta)

        obs = self._make_obs()
        info = {"theta": self._theta,
            "p": self._p,}
        return [obs, info]

    def _step_physics(self, action, dt):
        # we need to convert action from an array into a scalar
        u = float(np.clip(action[0], self.min_torque, self.max_torque))
        theta_new, p_new = self.integrator(self._theta, self._p, u, dt)
        self._theta = wrap_to_pi(theta_new)
        self._p = p_new
        new_cos_theta = np.cos(theta_new)
        new_sin_theta = np.sin(theta_new)
        self._cos_theta = new_cos_theta
        self._sin_theta = new_sin_theta

        theta_dt = p_new / self.I
        # changed reward from default to penalize angular velocity less
        phi = wrap_to_pi(theta_new - self._goal)
        reward = -(phi ** 2 + 0.01 * (theta_dt ** 2) + 0.0001 * u ** 2)
        if self.reward_version == "v1":
            # Holding bonus: paid per step for being inside the eps-ball about
            # upright, so reaching AND staying beats parking just outside it.
            dist = np.sqrt(phi ** 2 + theta_dt ** 2)
            if dist < self.hold_eps:
                reward += self.hold_bonus
        # need to decide if agent gets access to momentum or something else
        # maybe just give it momentum for now



        if self.q_only:
            self._push_measurement(theta_new)
        next_obs = self._make_obs()
        terminated = False
        truncated = False
        actual_dt_used = dt

        info = {"cos theta": self._cos_theta, 
                "sin_theta": self._sin_theta,
                "p": self._p,}
        
        return [next_obs, reward, terminated, truncated, info, actual_dt_used]



    # Need to make a decision about how momentum is handled
    # does agent get p or does it get an estimate 
    # we will allow the agent to access momentum in this preliminary trial, although this is certainly not a fair comparison to phast
    # in the q-only setting

    def force(self, theta, p, u):
        return -self.params.m * self.params.g * self.params.L * np.sin(theta) + u - self.compute_damping(theta) * (p / self.I)


    def diffusion(self, p):
        return self.sigma_gust + self.sigma_turb * abs(p / self.I)
        

    def integrator(self, theta, p, u, dt):
        # we integrate using a stochastic heun method 
        # initialize randomness
        xi = self._np_random.normal(0.0, 1)
        dW =  np.sqrt(dt) * xi
        # starting point
        f_n = self.force(theta, p, u)
        s_n = self.diffusion(p)

        # predictor, uses Euler step to this point
        theta_tilde = theta + (p / self.I) * dt    
        p_tilde = p + f_n * dt + s_n * dW    

        # corrector step 
        f_tilde = self.force(theta_tilde, p_tilde, u)
        s_tilde = self.diffusion(p_tilde)
        theta_new = theta + 0.5 * ((p / self.I) + (p_tilde / self.I)) * dt
        p_new = p + 0.5 * (f_n + f_tilde) * dt + 0.5 * (s_n + s_tilde) * dW
        return [theta_new , p_new]





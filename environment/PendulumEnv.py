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


class PendulumEnv(ContinuousEnv):
    metadata = {"render_modes": []}

    def __init__(self, *, dt: float = 0.02, max_torque: float = 20.0, min_torque: float = -20.0,
                params: PendulumParams, max_steps: int = 200, q_only: bool = False,
                obs_window: int = 4, meas_noise_std: float = 0.0):
        super().__init__(dt=dt, max_steps=int(max_steps))
        self.dt = float(dt)
        self.max_torque = float(max_torque)
        self.min_torque = float(min_torque)
        self.params = params
        self.b_base = self.params.b_base
        self.b_amplitude = self.params.b_amplitude

        # Observation mode.
        #   q_only=False -> full state [cos(theta), sin(theta), p].
        #   q_only=True  -> PHAST-style position-only sensing: a window of the
        #                   last `obs_window` (optionally noisy) position
        #                   measurements [cos, sin], with momentum hidden. The
        #                   agent must infer velocity from the position history.
        self.q_only = bool(q_only)
        self.obs_window = int(obs_window)
        self.meas_noise_std = float(meas_noise_std)
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
        """A (cos, sin) position measurement of `theta`, with optional Gaussian
        noise added to the angle (q-only sensing)."""
        if self.meas_noise_std > 0.0:
            theta = theta + self._np_random.normal(0.0, self.meas_noise_std)
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
        # Start near the upright equilibrium (theta = pi) with a small random
        # angular velocity, matching the PHAST Casimir control initialization
        # (theta ~ pi + N(0, 0.3), omega ~ N(0, 0.5)). This makes it a
        # stabilization/regulation task about the inverted position.
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
        reward = -((wrap_to_pi(theta_new - self._goal)) ** 2 + 0.01 * (theta_dt ** 2) + 0.0001 * u ** 2) 
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
    def integrator(self, theta, p, u, dt):
        # integrate one time step with strang splitting
        m, L, g = self.params.m, self.params.L, self.params.g
        # step 1: half damping at current theta
        D = self.compute_damping(theta)
        v = p / self.I
        p = p - 0.5 * dt * D * v
        #step 2: leapfrog 
        p_half = p - 0.5 * dt * m * g * L * np.sin(theta) + 0.5 * dt * u
        # full position update
        theta_new = theta + dt * p_half / self.I
        # half momentum kick at new position
        p = p_half - 0.5 * dt * m * g * L * np.sin(theta_new) + 0.5 * dt * u

        # step 3: half damping at new theta
        D_new = self.compute_damping(theta_new)
        v = p / self.I
        p_new = p - 0.5 * dt * D_new * v

        return [theta_new, p_new]
    


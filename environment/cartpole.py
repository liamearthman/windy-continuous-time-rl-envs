"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

## I am trying to make it a windy cartpole that is usable on ct-rl

import math
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from .base import ContinuousEnv


class CartPoleEnv(ContinuousEnv):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards
    Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

    If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |

    ## Vectorized environment

    To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

    ```python
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs
    CartPoleVectorEnv(CartPole-v1, num_envs=3)
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> envs
    SyncVectorEnv(CartPole-v1, num_envs=3)

    ```

    ## Version History
    * v1: `max_time_steps` raised to 500.
        - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: str | None = None, dt = 0.02, max_steps = 1000,
        q_only: bool = True, obs_window: int = 16,
        sigma_gust: float = 0.002, sigma_turb: float = 0.001, smooth_eps: float = 1e-3,
    ):
        super().__init__(dt=dt, max_steps=int(max_steps))
        self._sutton_barto_reward = sutton_barto_reward

        # Wind, same model as WindyPendulumEnv: the diffusion applied to the pole's
        # angular momentum is s = sigma_gust + sigma_turb * |theta_dot|, so a fast
        # swing is noisier than a slow one. sigma_gust = sigma_turb = 0 recovers the
        # deterministic cart-pole. smooth_eps rounds off |theta_dot| at the origin
        # (sqrt(thdot^2 + smooth_eps^2)) so the diffusion stays differentiable there.
        #
        # These defaults are NOT the pendulum's (0.1 / 0.05), and must not be copied
        # across. sigma is a kick to MOMENTUM, and what destabilises the pole is
        # ANGULAR VELOCITY, so the physical effect is scaled by the inverse mass
        # matrix. Here d(theta_dot)/d(p_theta) = (mc+mp)/det = 32.2, against 1/I = 1
        # for the pendulum -- the pole is 10x lighter with l^2 4x smaller, so the
        # same sigma delivers ~30x the angular-velocity noise. Worse, the noise
        # lands on p_theta while the force lands on p_x and reaches the pole only
        # through the mass-matrix coupling (gain 1.46), a 22x channel asymmetry.
        # At sigma_gust=0.1 the task is unsolvable: random actions survive 12.5
        # steps against 13 for zero force, i.e. control does not affect the outcome.
        # 0.002 restores the pendulum's ~0.010 rad/s per-step velocity noise; 0.02
        # is a reasonable "hard" setting for a robustness sweep.
        self.sigma_gust = float(sigma_gust)
        self.sigma_turb = float(sigma_turb)
        self.smooth_eps = float(smooth_eps)
        self._last_wind_impulse = 0.0   # diagnostic; set by integrator() each step

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.max_force = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.dt = dt
        self.max_steps = max_steps
        self.I = 1 # may need to change this 


        self.q_only = bool(q_only)
        self.obs_window = int(obs_window)
        self._obs_buf = np.zeros((self.obs_window, 3), dtype=np.float32)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32
        )        
        if self.q_only:
            # q-only sensing: a window of `obs_window` position measurements
            # [cos(theta), sin(theta), x] per frame (velocities hidden). Bounds
            # must match _measure's order: cos, sin in [-1, 1]; x in +/- x_threshold*2.
            obs_high = np.tile(
                [1.0, 1.0, self.x_threshold * 2], self.obs_window
            ).astype(np.float32)
            self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None


    def _measure(self, theta, x):
        """A (cos, sin) position measurement of `theta` (q-only sensing)."""
        return np.float32(np.cos(theta)), np.float32(np.sin(theta)), x

    def _push_measurement(self, theta, x):
        """Append a new position measurement to the sliding window (oldest first)."""
        self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
        self._obs_buf[-1] = self._measure(theta, x)

    def _make_obs(self):
        """Build the observation for the current mode."""
        if self.q_only:
            return self._obs_buf.reshape(-1).astype(np.float32)
        return self.state


    # def integrator(self, action, dt):
    #     x, x_dot, theta, theta_dot = self.state
    #     force = float(np.clip(np.asarray(action).reshape(-1)[0], -self.max_force, self.max_force))
    #     costheta = np.cos(theta)
    #     sintheta = np.sin(theta)
    #     temp = (
    #         force + self.polemass_length * np.square(theta_dot) * sintheta
    #     ) / self.total_mass
    #     thetaacc = (self.gravity * sintheta - costheta * temp) / (
    #         self.length
    #         * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
    #     )
    #     xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

    #     # if self.kinematics_integrator == "euler":
    #     #     x = x + self.dt * x_dot
    #     #     x_dot = x_dot + self.dt * xacc
    #     #     theta = theta + self.dt * theta_dot
    #     #     theta_dot = theta_dot + self.dt * thetaacc
    #     # else:  # semi-implicit euler
    #     x_dot = x_dot + dt * xacc
    #     x = x + dt * x_dot
    #     theta_dot = theta_dot + dt * thetaacc
    #     theta = theta + dt * theta_dot
    #     return x, x_dot, theta, theta_dot

    # ---------------------------------------------------------------------
    # State layout is (x, theta, p_x, p_theta) -- POSITIONS then MOMENTA, not
    # gym's (x, x_dot, theta, theta_dot). The mass matrix couples the two
    # velocities, so momenta are not just scaled velocities:
    #     M(theta) = [[ mc+mp,          mp*l*cos(theta) ],
    #                 [ mp*l*cos(theta), (4/3)*mp*l^2   ]]
    #     p = M(theta) @ [x_dot, theta_dot]
    # _velocities applies M^-1, _momenta applies M; they are exact inverses.
    # ---------------------------------------------------------------------

    def _velocities(self, theta, px, pth):
        mp, mc, l = self.masspole, self.masscart, self.length
        mtot, c = mc + mp, np.cos(theta)
        det = mtot * (4.0/3.0) * mp * l**2 - (mp * l * c)**2
        a, b, d = (4.0/3.0) * mp * l**2, -mp * l * c, mtot
        return (a*px + b*pth) / det, (b*px + d*pth) / det

    def _momenta(self, theta, xdot, thdot):
        """Inverse of _velocities: (x_dot, theta_dot) -> (p_x, p_theta)."""
        mp, mc, l = self.masspole, self.masscart, self.length
        mtot, c = mc + mp, np.cos(theta)
        return (mtot * xdot + mp * l * c * thdot,
                mp * l * c * xdot + (4.0/3.0) * mp * l**2 * thdot)

    def _drift(self, z, F):
        x, theta, px, pth = z
        mp, l, g = self.masspole, self.length, self.gravity
        s = np.sin(theta)
        xdot, thdot = self._velocities(theta, px, pth)
        return np.array([xdot, thdot, F, mp*g*l*s - mp*l*s*xdot*thdot])

    def diffusion(self, z):
        _, thdot = self._velocities(z[1], z[2], z[3])
        v = np.sqrt(thdot**2 + self.smooth_eps**2)       # smooth |thdot|
        return self.sigma_gust + self.sigma_turb * v

    def integrator(self, action, dt):
        # Draws from self._np_random -- the per-env stream seeded by reset(seed=...)
        # -- rather than taking an rng argument, so the wind realization is fixed by
        # the env seed alone and a rollout is reproducible from it. Same convention
        # as WindyPendulumEnv.integrator.
        z = np.asarray(self.state, dtype=float)
        F = float(np.clip(np.asarray(action).reshape(-1)[0], -self.max_force, self.max_force))

        dW = np.sqrt(dt) * self._np_random.standard_normal()
        G  = np.array([0.0, 0.0, 0.0, 1.0])

        s0 = self.diffusion(z)
        f0 = self._drift(z, F)
        z_t = z + f0*dt + s0*dW*G                        # predictor: same sigma
        s1 = self.diffusion(z_t)
        f1 = self._drift(z_t, F)

        # Record the noise impulse actually delivered to p_theta this step, so a
        # caller can report the REALIZED wind rather than only the sigmas it was
        # configured with. Diagnostic only -- nothing in the dynamics reads it.
        self._last_wind_impulse = float(0.5*(s0 + s1)*dW)

        return tuple(z + 0.5*(f0 + f1)*dt + 0.5*(s0 + s1)*dW*G)

    def wind_delta_thetadot(self):
        """Last step's wind impulse, expressed as the angular velocity it caused.

        The impulse lands on p_theta, and d(theta_dot)/d(p_theta) = (mc+mp)/det
        is large here (~32), so a small momentum kick is a much larger velocity
        change. Reporting rad/s makes the wind directly comparable to the pole's
        own motion and to what the force can do -- momentum units hide that.
        """
        _, dthd = self._velocities(self.state[1], 0.0, self._last_wind_impulse)
        return float(dthd)

    def wind_sigma_delta_thetadot(self, dt=None):
        """The 1-sigma scale of the above, for reference against the realized value."""
        dt = self.dt if dt is None else dt
        _, dthd = self._velocities(self.state[1], 0.0,
                                   self.diffusion(np.asarray(self.state, dtype=float))
                                   * math.sqrt(dt))
        return abs(float(dthd))

    def _step_physics(self, action, dt):
        assert self.state is not None, "Call reset before using step method."


        # moved integrating dynamics to a separate function
        x, theta, px, pth = self.integrator(action, dt)
        self.state = np.array((x, theta, px, pth), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        if self.q_only:
            self._push_measurement(theta, x)
        next_obs = self._make_obs()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return [next_obs, reward, terminated, False, {}, dt]

    def _reset_physics(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options,
            -0.05,
            0.05,  # default low
        )  # default high
        # Sample the DOCUMENTED starting state -- (x, x_dot, theta, theta_dot) all
        # uniform in (low, high) -- then convert the velocities to momenta, because
        # the state vector this env integrates is (x, theta, p_x, p_theta). Storing
        # the raw uniform draw would silently reinterpret x_dot as theta and
        # theta_dot as p_x.
        x0, xdot0, theta0, thdot0 = self._np_random.uniform(low=low, high=high, size=(4,))
        px0, pth0 = self._momenta(theta0, xdot0, thdot0)
        self.state = np.array((x0, theta0, px0, pth0), dtype=np.float64)
        self.steps_beyond_terminated = None

        if self.q_only:
            # Prime the window with the initial measurement (so the implied
            # velocity starts at zero), matching PendulumEnv's q_hist = q0.repeat(W).
            self._obs_buf[:] = self._measure(theta0, x0)

        if self.render_mode == "human":
            self.render()
        return np.array(self._make_obs(), dtype=np.float32), {}

    # have not looked at rendering yet

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            # state is (x, theta, p_x, p_theta): the pole angle is index 1.
            # Index 2 is p_x, which is what this used to rotate by.
            coord = pygame.math.Vector2(coord).rotate_rad(-x[1])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        sutton_barto_reward: bool = False,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.surf = None

        self.steps_beyond_terminated = None

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward:
            reward = -np.array(terminated, dtype=np.float32)
        else:
            reward = np.ones_like(terminated, dtype=np.float32)

        # Reset all environments which terminated or were truncated in the last step
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.prev_done.sum())
        )
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = np.logical_or(terminated, truncated)

        return self.state.T.astype(np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # -0.05 and 0.05 is the default low and high bounds
        self.low, self.high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        )
        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.state.T.astype(np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make_vec("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screens is None:
            pygame.display.init()

            self.screens = [
                pygame.Surface((self.screen_width, self.screen_height))
                for _ in range(self.num_envs)
            ]

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            raise ValueError(
                "Cartpole's state is None, it probably hasn't be reset yet."
            )

        for x, screen in zip(self.state.T, self.screens, strict=True):
            assert isinstance(x, np.ndarray) and x.shape == (4,)

            self.surf = pygame.Surface((self.screen_width, self.screen_height))
            self.surf.fill((255, 255, 255))

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
            carty = 100  # TOP OF CART
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )

            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )

            gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

            self.surf = pygame.transform.flip(self.surf, False, True)
            screen.blit(self.surf, (0, 0))

        return [
            np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
            for screen in self.screens
        ]

    def close(self, **kwargs: Any):
        if self.screens is not None:
            import pygame

            pygame.quit()

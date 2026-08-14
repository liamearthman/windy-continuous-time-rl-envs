"""classic Acrobot task

## Description

    The Acrobot environment is based on Sutton's work in
    ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
    and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
    The system consists of two links connected linearly to form a chain, with one end of
    the chain fixed. The joint between the two links is actuated. The goal is to apply
    torques on the actuated joint to swing the free end of the linear chain above a
    given height while starting from the initial state of hanging downwards.

    As seen in the **Gif**: two blue links connected by two green joints. The joint in
    between the two links is actuated. The goal is to swing the free end of the outer-link
    to reach the target height (black horizontal line above system) by applying torque on
    the actuator.

    ## Action Space

    The action is discrete, deterministic, and represents the torque applied on the actuated
    joint between the two links.

    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | apply -1 torque to the actuated joint | torque (N m) |
    | 1   | apply 0 torque to the actuated joint  | torque (N m) |
    | 2   | apply 1 torque to the actuated joint  | torque (N m) |

    ## Observation Space

    The observation is a `ndarray` with shape `(6,)` that provides information about the
    two rotational joint angles as well as their angular velocities:

    | Num | Observation                  | Min                 | Max               |
    |-----|------------------------------|---------------------|-------------------|
    | 0   | Cosine of `theta1`           | -1                  | 1                 |
    | 1   | Sine of `theta1`             | -1                  | 1                 |
    | 2   | Cosine of `theta2`           | -1                  | 1                 |
    | 3   | Sine of `theta2`             | -1                  | 1                 |
    | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    where
    - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards.
    - `theta2` is ***relative to the angle of the first link.***
        An angle of 0 corresponds to having the same angle between the two links.

    The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
    A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.

    ## Rewards

    The goal is to have the free end reach a designated target height in as few steps as possible,
    and as such all steps that do not reach the goal incur a reward of -1.
    Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

    ## Starting State

    Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
    uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: The free end reaches the target height, which is constructed as:
    `-cos(theta1) - cos(theta2 + theta1) > 1.0`
    2. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Acrobot only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make('Acrobot-v1', render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<AcrobotEnv<Acrobot-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.2, "high": 0.2})  # default low=-0.1, high=0.1
    (array([ 0.997341  ,  0.07287608,  0.9841162 , -0.17752565, -0.11185605,
           -0.12625128], dtype=float32), {})

    ```

    By default, the dynamics of the acrobot follow those described in Sutton and Barto's book
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/11/node4.html).
    However, a `book_or_nips` parameter can be modified to change the pendulum dynamics to those described
    in the original [NeurIPS paper](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html).

    ```python
    # To change the dynamics as described above
    env.unwrapped.book_or_nips = 'nips'
    ```

    See the following note for details:

    > The dynamics equations were missing some terms in the NIPS paper which are present in the book.
      R. Sutton confirmed in personal correspondence that the experimental results shown in the paper and the book were
      generated with the equations shown in the book. However, there is the option to run the domain with the paper equations
      by setting `book_or_nips = 'nips'`

    ## Version History

    - v1: Maximum number of steps increased from 200 to 500. The observation space for v0 provided direct readings of
    `theta1` and `theta2` in radians, having a range of `[-pi, pi]`. The v1 observation space as described here provides the
    sine and cosine of each angle instead.
    - v0: Initial versions release

    ## References
    - Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding.
        In D. Touretzky, M. C. Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing Systems (Vol. 8).
        MIT Press. https://proceedings.neurips.cc/paper/1995/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf
    - Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.

NOTE: the sections above describe the CLASSIC gymnasium acrobot. This module is the
continuous-time stochastic port-Hamiltonian version -- state is (theta1, theta2, p1, p2),
the action is a continuous torque Box(-torque_limit, torque_limit) rather than Discrete(3),
and integration is stochastic Heun at the env dt rather than RK4 over a fixed 0.2 s step.
"""


# this version of acrobot uses pytorch since it may be more useful for when we try to do phast comparisons, I am more comfortable with numpy so that is the primary 
# version I used


import math

import numpy as np
from numpy import cos, pi, sin
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from .base import ContinuousEnv



__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py



"""Windy Acrobot: stochastic port-Hamiltonian acrobot, Stratonovich / stochastic Heun."""


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta), np.cos(theta))

class AcrobotHamiltonian:
    """Analytic Hamiltonian + damping for the Acrobot (2-link underactuated pendulum).

    Plain class, not nn.Module: nothing here is learned or differentiated through,
    and subclassing nn.Module without calling its __init__ is a latent trap.
    torchacrobot.py is the differentiable variant.

    Coordinate convention:
      th1: shoulder angle
      th2: elbow angle relative to link 1
    """

    def __init__(
        self,
        m1: float = 1.0, m2: float = 1.0,
        l1: float = 1.0, l2: float = 1.0,
        lc1: float = 0.5, lc2: float = 0.5,
        I1: float = 1.0, I2: float = 1.0,
        g: float = 9.81,
        b1: float = 0.05, b2: float = 0.05,
    ):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = g
        self.b1, self.b2 = b1, b2
        # q = [costh1, sinth1, costh2, sinth2]
        # th2 = arcos(cos(th2))

    def mass_matrix(self, q: np.array):
        th2 = q[1]
        cos2 = np.cos(th2)
        d1 = (
            self.m1 * self.lc1**2
            + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * cos2)
            + self.I1 + self.I2
        )
        d2 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * cos2) + self.I2
        d3 = self.m2 * self.lc2**2 + self.I2
        return np.array([[d1,d2], [d2,d3]])

    def mass_matrix_dq2(self, q: np.ndarray) -> np.ndarray:
        """dM/dtheta2 -- M depends on q only through cos(theta2).

        d1 carries 2*l1*lc2*cos2 so its derivative is 2*dm; d2 carries one such term
        so its derivative is dm; d3 is constant.
        """
        s2 = np.sin(q[1])
        dm = -self.m2 * self.l1 * self.lc2 * s2
        return np.array([[2.0 * dm, dm], [dm, 0.0]])

    def grad_V(self, q: np.ndarray) -> np.ndarray:
        th1, th2 = q[0], q[1]
        dV1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.sin(th1) \
              + self.m2 * self.lc2 * self.g * np.sin(th1 + th2)
        dV2 = self.m2 * self.lc2 * self.g * np.sin(th1 + th2)
        return np.array([dV1, dV2])    

    def velocity(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        M = self.mass_matrix(q)
        return np.linalg.solve(M, p)

    def D(self, q: np.ndarray) -> np.ndarray:
        return np.diag([self.b1, self.b2])

    def grad_H(self, q: np.ndarray, p: np.ndarray):
        qdot = self.velocity (q, p)
        dM_dq2 = self.mass_matrix_dq2(q)
        dH_dtheta2 = -0.5 * (qdot @ dM_dq2 @ qdot)
        dV = self.grad_V(q)
        dH_dq = np.array([dV[0], dV[1] + dH_dtheta2])
        return dH_dq, qdot

    def potential(self, q: np.ndarray) -> float:
        th1, th2 = q[0], q[1]
        V1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * (1.0 - np.cos(th1))
        V2 = self.m2 * self.lc2 * self.g * (1.0 - np.cos(th1 + th2))
        return V1 + V2

    def energy(self, q: np.ndarray, p: np.ndarray) -> float:
        qdot = self.velocity(q, p)
        T = 0.5 * np.dot(p, qdot)
        return T + self.potential(q)


def acrobot_drift(ham: AcrobotHamiltonian, x: np.ndarray, u: float) -> np.ndarray:
    """f(x,u), x = (q1,q2,p1,p2). Deterministic port-Hamiltonian vector field."""
    q, p = x[:2], x[2:]
    dH_dq, dH_dp = ham.grad_H(q, p)
    dq = dH_dp
    diss = ham.D(q) @ dq
    dp = -dH_dq - diss
    dp = dp.copy()
    dp[1] += u   # B = [0,1]^T, joint-2 torque only
    return np.concatenate([dq, dp])


def acrobot_diffusion(ham: AcrobotHamiltonian, x: np.ndarray, sigma_gust: float, sigma_turb: float) -> np.ndarray:
    """G(x): scalar noise channel entering as a disturbance torque on the UNACTUATED joint (p1)."""
    q, p = x[:2], x[2:]
    qdot1 = ham.velocity(q, p)[0]
    sigma = sigma_gust + sigma_turb * abs(qdot1)
    G = np.zeros_like(x)
    G[2] = sigma   # p1 slot only
    return G


class WindyAcrobotEnv(ContinuousEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }


    def __init__(self,dt: float = 0.02, torque_limit: float = 5.0, sigma_gust : float = 0.01,
                 sigma_turb: float = 0.01, max_steps: int = 1000, q_only: bool = True,
                 obs_window: int = 16, render_mode: str | None = None,
                 reward_version: str = "v0", goal_height: float = 1.0,
                 hold_bonus: float = 1.0, hold_rate: float = 1.0,):
        # MUST come first: ContinuousEnv.__init__ assigns self.dt and self.max_steps,
        # so calling it at the end (with no args) silently reset max_steps to None.
        super().__init__(dt=dt, max_steps=int(max_steps))
        self.ham = AcrobotHamiltonian()
        self.dt = dt
        self.torque_limit = torque_limit
        self.sigma_turb = sigma_turb
        self.sigma_gust = sigma_gust
        self.max_steps = max_steps
        self.q_only = q_only
        # Position-only sensing needs HISTORY: a single frame of angles encodes no
        # velocity at all, so a feedforward policy could never recover the rates.
        # The window holds the last `obs_window` measurements
        # [cos th1, sin th1, cos th2, sin th2], oldest first, newest last -- the same
        # convention as WindyPendulumEnv (25 frames x 2) and the CT CartPoleEnv
        # (16 frames x 3). Velocity is inferable by differencing across frames.
        self.obs_window = int(obs_window)
        self._obs_buf = np.zeros((self.obs_window, 4), dtype=np.float32)
        self.render_mode = render_mode
        # Reward.
        #   "v0" -- the classic sparse -1 per step, 0 on reaching the goal. Standard,
        #           but MEASURED to be unlearnable here: random exploration never sees
        #           the goal (0/30 episodes, best tip height -1.56 against a +1.0 bar),
        #           so the return is a flat -max_steps with no gradient anywhere.
        #   "v1" -- dense shaping on how far the tip is below the bar, plus small rate
        #           and effort penalties and a hold bonus once above it. Same structure
        #           as the windy pendulum's v1.
        self.reward_version = str(reward_version)
        self.goal_height = float(goal_height)
        self.hold_bonus = float(hold_bonus)
        self.hold_rate = float(hold_rate)
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None
        self.action_space = spaces.Box(low =-torque_limit, high= torque_limit, shape=(1,), dtype=np.float32)

        # q_only=True is PHAST-style position-only sensing: a WINDOW of
        # [cos th1, sin th1, cos th2, sin th2] frames with both rates hidden.
        # q_only=False is the single-frame full state including the two rates.
        if self.q_only:
            # cos/sin are always in [-1, 1], so the box is exact rather than inf.
            high = np.ones(4 * self.obs_window, dtype=np.float32)
        else:
            high = np.full(6, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)


    def _reset_physics(self, *, seed: int | None = None, options: dict | None = None):
        # No super().reset() here: ContinuousEnv.reset() is what CALLS this hook, so
        # re-entering it recurses forever. The base has already seeded _np_random.
        low, high = -0.1, 0.1
        if options:
            low = float(options.get("low", low))
            high = float(options.get("high", high))
        q = self._np_random.uniform(low, high, size=2)
        qdot = self._np_random.uniform(low, high, size=2)
        p = self.ham.mass_matrix(q) @ qdot
        self.state = np.concatenate([q, p])
        if self.q_only:
            # Pad every frame with the initial measurement, so the implied velocity
            # starts at zero -- same convention as the pendulum and cart-pole.
            self._obs_buf[:] = self._measure(q[0], q[1])
        if self.render_mode == "human":
            self.render()
        return [self._get_ob(), {"theta1": q[0], "theta2": q[1],
                                 "p1": p[0], "p2": p[1]}]

    def _step_physics(self, action, dt):
        # ContinuousEnv passes the dt for THIS step (it may vary under irregular
        # sampling), so integrate with it rather than with self.dt.
        x = self.state
        u = float(np.clip(np.asarray(action).reshape(-1)[0],
                          -self.torque_limit, self.torque_limit))

        dW = np.sqrt(dt) * self._np_random.normal()   # single shared Brownian increment

        f0 = acrobot_drift(self.ham, x, u)
        g0 = acrobot_diffusion(self.ham, x, self.sigma_gust, self.sigma_turb)
        x_pred = x + f0 * dt + g0 * dW

        f1 = acrobot_drift(self.ham, x_pred, u)
        g1 = acrobot_diffusion(self.ham, x_pred, self.sigma_gust, self.sigma_turb)
        x_next = x + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dW

        x_next = x_next.copy()
        x_next[0] = wrap_angle(x_next[0])
        x_next[1] = wrap_angle(x_next[1])
        self.state = x_next

        terminated = self._terminal()
        tip = float(-np.cos(self.state[0]) - np.cos(self.state[0] + self.state[1]))
        if self.reward_version == "v1":
            qdot = self.ham.velocity(self.state[:2], self.state[2:])
            shortfall = max(0.0, self.goal_height - tip)
            # NOTE the rate penalty is applied ONLY at the goal. Swing-up is an
            # energy-pumping problem -- the arm has to build up speed before it can
            # gain height -- so a global penalty on |qdot| actively fights the task.
            # Penalising rates only once the tip is up rewards arriving AND settling.
            reward = -(shortfall ** 2) - 0.001 * u * u
            if terminated:
                reward += self.hold_bonus - 0.01 * float(qdot[0] ** 2 + qdot[1] ** 2)
        else:
            reward = -1.0 if not terminated else 0.0
        if self.q_only:
            self._push_measurement(self.state[0], self.state[1])
        if self.render_mode == "human":
            self.render()
        # ContinuousEnv expects six values, the last being the dt actually used.
        info = {"tip_height": float(-np.cos(self.state[0])
                                    - np.cos(self.state[0] + self.state[1]))}
        return [self._get_ob(), reward, terminated, False, info, dt]

    def _terminal(self) -> bool:
        q = self.state[:2]
        return bool(-np.cos(q[0]) - np.cos(q[0] + q[1]) > 1.0)

    def _measure(self, th1: float, th2: float):
        """One position measurement: (cos th1, sin th1, cos th2, sin th2)."""
        return (np.float32(np.cos(th1)), np.float32(np.sin(th1)),
                np.float32(np.cos(th2)), np.float32(np.sin(th2)))

    def _push_measurement(self, th1: float, th2: float) -> None:
        """Slide the window down one frame and write the newest measurement last."""
        self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
        self._obs_buf[-1] = self._measure(th1, th2)

    def _get_ob(self) -> np.ndarray:
        q, p = self.state[:2], self.state[2:]
        if self.q_only:
            # Flattened oldest-first window; rates are hidden and must be inferred
            # by differencing successive frames.
            return self._obs_buf.reshape(-1).astype(np.float32)
        qdot = self.ham.velocity(q, p)
        return np.array([np.cos(q[0]), np.sin(q[0]), np.cos(q[1]), np.sin(q[1]),
                         qdot[0], qdot[1]], dtype=np.float32)
    # havent looked at anything below here

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
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        xys = np.array([[0, 0], p1])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        for (x, y), th, llen in zip(xys, thetas, link_lengths, strict=True):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

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


def wrap(x, m, M):
    """Wraps `x` so m <= x <= M; but unlike `bound()` which
    truncates, `wrap()` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature `dy = derivs(yi)`
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float64)
    else:
        yout = np.zeros((len(t), Ny), np.float64)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]
